import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
import difflib
import gc

# ---------- Render SINGLE page to image ----------
def render_page(doc, i, dpi=150):
    page = doc[i]
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)
    return img

# ---------- Extract text from PDF ----------
@st.cache_data
def extract_pdf_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return [page.get_text() for page in doc]

# ---------- Highlight differences ----------
def highlight_differences_clustered(img1, img2, padding=30, min_area=100):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]

    # # OPTIONAL: downscale for speed
    # img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
    # img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)

    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        if w_box * h_box >= min_area:
            boxes.append([x - padding, y - padding, x + w_box + padding, y + h_box + padding])

    def merge_overlapping(boxes):
        merged = []
        for box in boxes:
            x1, y1, x2, y2 = box
            merged_flag = False
            for i, m in enumerate(merged):
                mx1, my1, mx2, my2 = m
                if not (x2 < mx1 or x1 > mx2 or y2 < my1 or y1 > my2):
                    merged[i] = [min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2)]
                    merged_flag = True
                    break
            if not merged_flag:
                merged.append(box)

        changed = True
        while changed:
            changed = False
            new_merged = []
            for box in merged:
                x1, y1, x2, y2 = box
                merged_flag = False
                for i, m in enumerate(new_merged):
                    mx1, my1, mx2, my2 = m
                    if not (x2 < mx1 or x1 > mx2 or y2 < my1 or y1 > my2):
                        new_merged[i] = [min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2)]
                        merged_flag = True
                        changed = True
                        break
                if not merged_flag:
                    new_merged.append(box)
            merged = new_merged

        final_boxes = [[max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)] for x1, y1, x2, y2 in merged]
        return final_boxes

    clustered_boxes = merge_overlapping(boxes)

    img1_high = img1.copy()
    img2_high = img2.copy()

    has_diff = len(clustered_boxes) > 0
    for x1, y1, x2, y2 in clustered_boxes:
        cv2.rectangle(img1_high, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(img2_high, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img1_high, img2_high, has_diff, clustered_boxes

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide")
st.title("📄 PDF Diff")

col1, col2 = st.columns(2)
with col1:
    pdf1 = st.file_uploader("Upload PDF 1", type="pdf")
with col2:
    pdf2 = st.file_uploader("Upload PDF 2", type="pdf")

if pdf1 and pdf2:

    # Read once
    pdf1_bytes = pdf1.read()
    pdf2_bytes = pdf2.read()

    with st.spinner("Extracting text..."):
        text1 = extract_pdf_text(pdf1_bytes)
        text2 = extract_pdf_text(pdf2_bytes)

    # Open docs (no rendering yet)
    doc1 = fitz.open(stream=pdf1_bytes, filetype="pdf")
    doc2 = fitz.open(stream=pdf2_bytes, filetype="pdf")
    num_pages1 = len(doc1)
    num_pages2 = len(doc2)
    num_pages = min(len(doc1), len(doc2))

    mode = st.sidebar.radio(
        "Comparison Mode",
        ["Auto (Aligned Pages)", "Manual Page Selection", "Text Diff Only"]
    )

    # ---------- AUTO MODE ----------
    if mode == "Auto (Aligned Pages)":

        pages_with_diffs = [
            i + 1 for i in range(num_pages)
            if text1[i] != text2[i]
        ]

        if pages_with_diffs:
            st.sidebar.markdown("### 📌 Differences")
            selected_page = st.sidebar.selectbox(
                "Jump to page with differences",
                pages_with_diffs
            )

            page_idx = selected_page - 1

            with st.spinner(f"Rendering page {selected_page}..."):
                img1 = render_page(doc1, page_idx)
                img2 = render_page(doc2, page_idx)

                img1_high, img2_high, has_diff, boxes = highlight_differences_clustered(img1, img2)

                del img1, img2
                gc.collect()

            col1, col2 = st.columns(2)
            with col1:
                st.image(img1_high, caption=f"PDF 1 - Page {selected_page}", use_container_width=True)
            with col2:
                st.image(img2_high, caption=f"PDF 2 - Page {selected_page}", use_container_width=True)

        else:
            st.info("No visual differences detected.")

    # ---------- MANUAL MODE ----------
    elif mode == "Manual Page Selection": 

        st.sidebar.markdown("### 🔄 Manual Page Comparison")

        page1 = st.sidebar.number_input(
            "PDF 1 - Page",
            min_value=1,
            max_value=num_pages1,
            value=1
        )

        page2 = st.sidebar.number_input(
            "PDF 2 - Page",
            min_value=1,
            max_value=num_pages2,
            value=1
        )

        page1_idx = page1 - 1
        page2_idx = page2 - 1

        with st.spinner(f"Comparing Page {page1} vs Page {page2}..."):
            img1 = render_page(doc1, page1_idx)
            img2 = render_page(doc2, page2_idx)

            img1_high, img2_high, has_diff, boxes = highlight_differences_clustered(img1, img2)

            del img1, img2
            gc.collect()

        col1, col2 = st.columns(2)
        with col1:
            st.image(img1_high, caption=f"PDF 1 - Page {page1}", use_container_width=True)
        with col2:
            st.image(img2_high, caption=f"PDF 2 - Page {page2}", use_container_width=True)

    # ---------- TEXT DIFF ONLY MODE ----------
    elif mode == "Text Diff Only":

        st.sidebar.markdown("### 📝 Text Diff Navigation")

        # 🔄 NEW: Manual page selection (same as manual mode)
        page1 = st.sidebar.number_input(
            "PDF 1 - Page",
            min_value=1,
            max_value=num_pages1,
            value=1,
            key="text_page1"
        )

        page2 = st.sidebar.number_input(
            "PDF 2 - Page",
            min_value=1,
            max_value=num_pages2,
            value=1,
            key="text_page2"
        )

        page1_idx = page1 - 1
        page2_idx = page2 - 1

        # Use selected pages instead of aligned index
        page1_obj = doc1[page1_idx]
        page2_obj = doc2[page2_idx]

        text_lines1 = text1[page1_idx].splitlines()
        text_lines2 = text2[page2_idx].splitlines()

        diff = list(difflib.ndiff(text_lines1, text_lines2))

        # Extract changes
        removed = [line[2:] for line in diff if line.startswith("- ")]
        added = [line[2:] for line in diff if line.startswith("+ ")]

        # ---------- Highlight ----------
        page1_clean = fitz.open(stream=pdf1_bytes, filetype="pdf")[page1_idx]
        page2_clean = fitz.open(stream=pdf2_bytes, filetype="pdf")[page2_idx]

        for line in removed:
            if line.strip():
                areas = page1_clean.search_for(line)
                for rect in areas:
                    annot = page1_clean.add_highlight_annot(rect)
                    annot.set_colors(stroke=(1, 0, 0))  # red
                    annot.update()

        for line in added:
            if line.strip():
                areas = page2_clean.search_for(line)
                for rect in areas:
                    annot = page2_clean.add_highlight_annot(rect)
                    annot.set_colors(stroke=(0, 1, 0))  # green
                    annot.update()

        # ---------- Render ----------
        with st.spinner(f"Comparing Page {page1} vs Page {page2}..."):
            pix1 = page1_clean.get_pixmap()
            pix2 = page2_clean.get_pixmap()

            img1 = np.frombuffer(pix1.samples, dtype=np.uint8).reshape(pix1.height, pix1.width, pix1.n)
            img2 = np.frombuffer(pix2.samples, dtype=np.uint8).reshape(pix2.height, pix2.width, pix2.n)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img1, caption=f"PDF 1 - Page {page1} (Removed)", use_container_width=True)

        with col2:
            st.image(img2, caption=f"PDF 2 - Page {page2} (Added)", use_container_width=True)

        # ---------- Diff Output ----------
        st.markdown("---")
        st.subheader("📝 Diff Output")

        st.code("\n".join(diff), language="diff")

# Only show full diff in non-text-only modes
# if mode != "Text Diff Only":

#     st.markdown("---")
#     st.subheader("📝 Textual Differences")

#     for i in range(num_pages):
#         page_text1 = text1[i].splitlines()
#         page_text2 = text2[i].splitlines()

#         diff_lines = list(
#             difflib.unified_diff(
#                 page_text1,
#                 page_text2,
#                 lineterm='',
#                 fromfile=f'PDF1 Page {i+1}',
#                 tofile=f'PDF2 Page {i+1}'
#             )
#         )

#         if diff_lines:
#             with st.expander(f"Page {i+1} Text Differences"):
#                 st.text('\n'.join(diff_lines))
