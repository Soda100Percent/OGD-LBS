# filepath: /workspace/lao-lbs/src/lao_lbs/blender/image_io.py
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import bpy

# 修正导入路径
from ogd_lbs.utils.common import log, warn, fail, Timer


SUPPORTED_IMAGE_FORMATS = {"PNG", "WEBP"}


def get_image_format(args: Any) -> str:
    fmt = str(getattr(args, "image_format", "PNG") or "PNG").upper()

    if fmt not in SUPPORTED_IMAGE_FORMATS:
        warn(f"Unsupported image_format={fmt!r}; fallback to PNG.")
        return "PNG"

    return fmt


def get_image_extension(args: Any) -> str:
    fmt = get_image_format(args)

    if fmt == "WEBP":
        return ".webp"

    return ".png"


def sanitize_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def make_temp_capture_path(final_path: str | Path, *, pass_name: str) -> Path:
    """
    OpenGL / final render 的临时 PNG 捕获路径。

    设计原则：
        不让 Blender OpenGL 直接输出 WEBP。
        OpenGL 永远先捕获为 PNG。
        后续由 finalize_captured_png() 转为 PNG / lossless WEBP。
    """
    final_path = Path(final_path)
    tag = sanitize_tag(pass_name)
    pid = os.getpid()

    return final_path.with_name(
        f"__tmp_capture_{final_path.stem}_{tag}_{pid}.png"
    )


def make_temp_png_path(final_path: str | Path, *, suffix_tag: str) -> Path:
    final_path = Path(final_path)
    tag = sanitize_tag(suffix_tag)
    pid = os.getpid()

    return final_path.with_name(
        f"__tmp_{final_path.stem}_{tag}_{pid}.png"
    )


def detect_image_magic(path: str | Path) -> str:
    """
    根据文件头判断真实图像格式，而不是相信扩展名。
    """
    path = Path(path)

    if not path.exists():
        return "MISSING"

    try:
        with path.open("rb") as f:
            head = f.read(16)
    except Exception:
        return "UNKNOWN"

    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"

    if len(head) >= 12 and head[0:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"

    return "UNKNOWN"


def validate_image_magic(
    path: str | Path,
    expected_format: str,
    *,
    strict: bool = True,
) -> bool:
    with Timer(f"[T5] Magic Header Check: {Path(path).name}"):
        path = Path(path)
        expected = expected_format.upper()
        actual = detect_image_magic(path)

        if actual == expected:
            log(f"Validated image format: {path.name} is real {actual}.")
            return True

        msg = (
            f"Image format mismatch: path={path}, "
            f"expected={expected}, actual_magic={actual}."
        )

        if strict:
            fail(msg)

        warn(msg)
        return False


def require_cwebp() -> str:
    """
    WEBP 无损输出依赖 cwebp。
    """
    exe = shutil.which("cwebp")

    if not exe:
        fail(
            "Requested --image_format WEBP, but cwebp was not found in PATH.\n"
            "Install it first, for example:\n"
            "    sudo apt-get update && sudo apt-get install -y webp\n"
            "or:\n"
            "    conda install -c conda-forge libwebp\n"
            "Then verify:\n"
            "    which cwebp && cwebp -version"
        )

    return exe


def configure_scene_for_png_capture(scene: bpy.types.Scene) -> None:
    """
    配置 Blender 先稳定输出 PNG/RGBA。

    这是 capture 设置，不代表最终格式。
    最终 WEBP 由 cwebp 从 PNG 编码得到。
    """
    scene.render.film_transparent = True
    scene.render.use_file_extension = False
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    try:
        # 【性能优化】将临时捕获 PNG 的压缩率降为 0，极速写入磁盘
        scene.render.image_settings.compression = 0
    except Exception:
        pass


def configure_scene_for_final_format(
    scene: bpy.types.Scene,
    image_format: str,
) -> None:
    """
    记录最终格式意图。

    注意：
        OpenGL pass 不直接依赖 Blender 写 WEBP。
        它总是先 PNG capture，然后 cwebp 转码。
    """
    fmt = image_format.upper()

    scene.render.film_transparent = True
    scene.render.use_file_extension = False
    scene.render.resolution_percentage = 100

    if fmt == "WEBP":
        # 尽量配置 Blender 的 WEBP 状态，但不依赖它保证最终输出。
        try:
            scene.render.image_settings.file_format = "WEBP"
            scene.render.image_settings.color_mode = "RGBA"
            scene.render.image_settings.quality = 100

            if hasattr(scene.render.image_settings, "use_lossless"):
                try:
                    scene.render.image_settings.use_lossless = True
                except Exception:
                    pass
        except Exception as exc:
            warn(f"Failed to configure Blender WEBP settings: {exc}")

        require_cwebp()
        return

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    try:
        scene.render.image_settings.compression = 100
    except Exception:
        pass


def encode_png_to_lossless_webp(
    png_path: str | Path,
    webp_path: str | Path,
    *,
    remove_input: bool = False,
) -> None:
    """
    使用 cwebp 生成真正无损 WebP。
    """
    png_path = Path(png_path)
    webp_path = Path(webp_path)

    if not png_path.exists():
        fail(f"PNG input for WEBP encoding not found: {png_path}")

    cwebp = require_cwebp()

    webp_path.parent.mkdir(parents=True, exist_ok=True)

    # 【性能优化】加入 -mt 多线程，降低 -m 和 -q 以换取数倍的速度提升
    cmd = [
        cwebp,
        "-quiet",
        "-mt",
        "-lossless",
        "-exact",
        "-m", "4",
        "-q", "50",
        str(png_path),
        "-o",
        str(webp_path),
    ]

    log(f"Encoding lossless WEBP: {' '.join(cmd)}")

    with Timer(f"[T4] WebP Encoding (cwebp): {webp_path.name}"):
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    if result.returncode != 0:
        fail(
            "cwebp failed.\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )

    if not webp_path.exists():
        fail(f"cwebp finished but output not found: {webp_path}")

    if remove_input:
        try:
            png_path.unlink()
        except Exception as exc:
            warn(f"Failed to remove temp PNG after WEBP encode: {png_path}: {exc}")


def save_image_object_as_png(
    image: bpy.types.Image,
    png_path: str | Path,
) -> None:
    """
    将 Blender Image datablock 保存为 PNG。
    """
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    image.filepath_raw = str(png_path)
    image.file_format = "PNG"
    image.save()

    if not png_path.exists():
        fail(f"Failed to save PNG image object: {png_path}")


def save_image_object_to_final(
    image: bpy.types.Image,
    final_path: str | Path,
    image_format: str,
) -> None:
    """
    保存 crop 后的 Blender Image 为最终格式。
    """
    final_path = Path(final_path)
    fmt = image_format.upper()

    final_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "WEBP":
        tmp_png = make_temp_png_path(final_path, suffix_tag="crop_for_webp")

        try:
            save_image_object_as_png(image, tmp_png)
            encode_png_to_lossless_webp(
                tmp_png,
                final_path,
                remove_input=True,
            )
        finally:
            if tmp_png.exists():
                try:
                    tmp_png.unlink()
                except Exception:
                    pass
        return

    save_image_object_as_png(image, final_path)


def finalize_captured_png(
    capture_png_path: str | Path,
    final_path: str | Path,
    image_format: str,
    *,
    remove_input: bool = True,
) -> None:
    """
    将 OpenGL capture PNG 转成最终输出格式。
    """
    capture_png_path = Path(capture_png_path)
    final_path = Path(final_path)
    fmt = image_format.upper()

    if not capture_png_path.exists():
        fail(f"Capture PNG not found: {capture_png_path}")

    final_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "WEBP":
        encode_png_to_lossless_webp(
            capture_png_path,
            final_path,
            remove_input=remove_input,
        )
        return

    # Default: PNG
    if capture_png_path.resolve() != final_path.resolve():
        shutil.copy2(capture_png_path, final_path)

    if (
        remove_input
        and capture_png_path.exists()
        and capture_png_path.resolve() != final_path.resolve()
    ):
        try:
            capture_png_path.unlink()
        except Exception as exc:
            warn(f"Failed to remove temp capture PNG: {capture_png_path}: {exc}")


def remove_stale_alternative_outputs(final_path: str | Path) -> None:
    """
    删除同 stem 的旧格式输出，避免你看到旧 PNG 后误判。
    """
    final_path = Path(final_path)
    final_suffix = final_path.suffix.lower()

    for suffix in [".png", ".webp"]:
        if suffix == final_suffix:
            continue

        alt = final_path.with_suffix(suffix)

        if alt.exists():
            try:
                warn(f"Removing stale alternative output: {alt}")
                alt.unlink()
            except Exception as exc:
                warn(f"Failed to remove stale alternative output: {alt}: {exc}")


def cleanup_temp_files_for_final(final_path: str | Path) -> None:
    """
    清理上次中断残留的当前 frame 临时文件。
    """
    final_path = Path(final_path)
    parent = final_path.parent

    if not parent.exists():
        return

    patterns = [
        f"__tmp_capture_{final_path.stem}_*.png",
        f"__tmp_{final_path.stem}_*.png",
    ]

    for pattern in patterns:
        for item in parent.glob(pattern):
            try:
                warn(f"Removing stale temp image: {item}")
                item.unlink()
            except Exception as exc:
                warn(f"Failed to remove stale temp image: {item}: {exc}")


def assert_final_output_exists(
    final_path: str | Path,
    image_format: str,
) -> None:
    """
    最终输出强校验。
    这是整个流水线中唯一保留的 Magic Header 校验点。
    """
    final_path = Path(final_path)
    fmt = image_format.upper()

    if not final_path.exists():
        fail(f"Final output was not created: {final_path}")

    validate_image_magic(final_path, fmt, strict=True)