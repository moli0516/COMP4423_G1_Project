"""Part C review UI backed by the real B chain, with explicit synthetic limits."""

import hashlib
import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deskcheck.application.app_config import AppConfig
from deskcheck.application.comparison_service import ComparisonService
from deskcheck.application.demo_pair import make_demo_pair
from deskcheck.application.image_pair_loader import ImagePairLoader
from deskcheck.application.presentation import comparison_rows, comparison_wipe
from deskcheck.application.result_renderer import ResultRenderer
from deskcheck.application.result_serializer import ResultSerializer
from deskcheck.detection.fake_color_detector import FakeColorDetector

st.set_page_config(page_title="DeskCheck | Jerry-C", layout="wide")
st.title("DeskCheck: before / after review")
st.caption("COMP4423 | Part C | One comparison service for the UI, CLI and B evaluation runner")
st.warning(
    "FAKE / SYNTHETIC ONLY. Exact-color shapes exercise real B software. "
    "No trained model is loaded; these results do not measure real-world accuracy."
)
source = st.radio(
    "Image source",
    ["Synthetic demo", "Upload synthetic images", "Inspect real photos (no AI)"],
    horizontal=True,
)
pair_id = st.text_input("Pair ID", "c-demo")
photo_only = source == "Inspect real photos (no AI)"
before_bytes = after_bytes = None
if source == "Synthetic demo":
    before_bytes, after_bytes = make_demo_pair()
else:
    left, right = st.columns(2)
    with left:
        before_upload = st.file_uploader("Before image", type=["jpg", "jpeg", "png"], key="before")
    with right:
        after_upload = st.file_uploader("After image", type=["jpg", "jpeg", "png"], key="after")
    if before_upload is not None:
        before_bytes = before_upload.getvalue()
    if after_upload is not None:
        after_bytes = after_upload.getvalue()
    if photo_only:
        st.info("Manual inspection only: no detector is called and no AI result is exported.")
    else:
        st.info(
            "Use the exact-color synthetic PNG fixtures. Ordinary photographs are not "
            "supported by the fake detector. No detections does not establish no change."
        )

try:
    config = AppConfig.load(ROOT / "configs" / "fake_app.json")
except (OSError, ValueError, TypeError) as error:
    st.session_state.pop("c_bundle", None)
    st.error(f"Configuration cannot be loaded: {error}")
    st.stop()

# Hash each input separately: concatenation delimiters can collide in arbitrary bytes.
parts = (
    hashlib.sha256(before_bytes or b"").hexdigest(),
    hashlib.sha256(after_bytes or b"").hexdigest(),
    pair_id,
    source,
    config.config_sha256,
)
fingerprint = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()
if st.session_state.get("c_fingerprint") != fingerprint:
    had_result = st.session_state.pop("c_bundle", None) is not None
    st.session_state["c_fingerprint"] = fingerprint
    if had_result:
        st.session_state["c_rerun_needed"] = True
if st.session_state.get("c_rerun_needed"):
    st.info("Inputs changed. Previous results and downloads were cleared; run again.")

ready = before_bytes is not None and after_bytes is not None and bool(pair_id.strip())
frames = None
if ready:
    try:
        if max(len(before_bytes), len(after_bytes)) > 32 * 1024 * 1024:
            raise ValueError("Each encoded image must be at most 32 MiB")
        frames = ImagePairLoader().load(before_bytes, after_bytes, pair_id)
    except (OSError, ValueError, TypeError) as error:
        st.session_state.pop("c_bundle", None)
        st.error(f"Check the selected images: {error}. Replace the image and try again.")

if frames is not None:
    with st.expander("Input previews (oriented originals)", expanded=photo_only):
        for column, frame, side in zip(st.columns(2), frames, ("Before", "After")):
            column.image(frame.array, caption=f"{side} | {frame.width} x {frame.height}")

if st.button("Compare synthetic inputs", disabled=frames is None or photo_only, type="primary"):
    st.session_state.pop("c_bundle", None)
    try:
        # Reuse the exact decoded pixels for inference, rendering and downloads.
        result = ComparisonService(detector=FakeColorDetector(), config=config).compare_frames(
            *frames, pair_id=pair_id
        )
        images = ResultRenderer().render(*frames, result)
        serializer = ResultSerializer()
        downloads = {
            "result.json": serializer.to_json(result).encode("utf-8"),
            "result.csv": serializer.to_csv(result).encode("utf-8"),
        }
        for side, image in zip(("before", "after"), images):
            output = io.BytesIO()
            image.save(output, format="PNG")
            downloads[f"annotated_{side}.png"] = output.getvalue()
        st.session_state["c_bundle"] = {
            "result": result,
            "frames": frames,
            "images": images,
            "downloads": downloads,
        }
        st.session_state.pop("c_rerun_needed", None)
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        st.error(f"Comparison failed: {error}. No result was published.")

bundle = st.session_state.get("c_bundle")
if bundle is not None and not photo_only:
    result = bundle["result"]
    count = len(result["records"])
    changes = sum(r["indicates_change"] for r in result["records"])
    one, two = st.columns(2)
    one.metric("Objects in comparison", count)
    two.metric("Changes", changes)
    if count == 0:
        st.info(
            "No target objects detected by the synthetic detector. This is not proof "
            "that a real scene has no changes."
        )
    elif changes == 0:
        st.success("All detected synthetic objects are unchanged under the configured rule.")
    view = st.radio(
        "Result view", ["Annotated", "Originals", "Before / after wipe"], horizontal=True
    )
    originals = tuple(Image.fromarray(f.array.copy()) for f in bundle["frames"])
    if view == "Before / after wipe":
        split = st.slider("Before image visible (%)", 0, 100, 50)
        st.image(comparison_wipe(*originals, split), caption="Before on left; after on right")
    else:
        shown = bundle["images"] if view == "Annotated" else originals
        for column, image, side in zip(st.columns(2), shown, ("Before", "After")):
            column.image(image, caption=side)
    changes_only = st.checkbox("Show changed objects only")
    rows = comparison_rows(result, changes_only=changes_only)
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "Display filtering never removes records from downloads. One object per class "
        "is the current project scope; the class links the before/after records."
    )
    if rows:
        labels = [f"{row['Object ID']} | {row['Object']} | {row['Status']}" for row in rows]
        chosen = st.selectbox("Object details", labels)
        chosen_row = rows[labels.index(chosen)]
        record = next(
            r for r in result["records"] if f"object-{r['class_id']}" == chosen_row["Object ID"]
        )
        with st.expander("Detection and movement evidence", expanded=True):
            st.text(
                f"Before: {record['before_detection_id']} | box={record['before_box']}\n"
                f"After: {record['after_detection_id']} | box={record['after_box']}"
            )
            st.caption(
                f"Center distance / image diagonal: {record['distance_norm']}. "
                f"Movement threshold: {record['movement_threshold']}; moved requires >, "
                "not equality. Added/removed use missing-side evidence."
            )
    with st.expander("Complete JSON and provenance"):
        st.json(result)
    st.subheader("Download complete comparison")
    names = [
        ("Download JSON", "result.json", "application/json"),
        ("Download CSV", "result.csv", "text/csv"),
        ("Download annotated before", "annotated_before.png", "image/png"),
        ("Download annotated after", "annotated_after.png", "image/png"),
    ]
    for column, (label, name, mime) in zip(st.columns(4), names):
        column.download_button(label, bundle["downloads"][name], file_name=name, mime=mime)
