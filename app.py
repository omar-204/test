from pathlib import Path
import streamlit as st
from utils.model_loader import load_local_model, model_summary
from utils.data_utils import discover_dataset, load_dataset, get_project_paths
from utils.styles import apply_styles

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="ML Project Workbench", page_icon=None, layout="wide", initial_sidebar_state="expanded")
apply_styles()

paths = get_project_paths(ROOT)
model, model_error = load_local_model(paths["model"])
dataset_path = discover_dataset(ROOT)
dataset = None
dataset_error = None
if dataset_path:
    try:
        dataset = load_dataset(str(dataset_path))
    except Exception as exc:
        dataset_error = str(exc)

with st.sidebar:
    st.markdown("### ML Project Workbench")
    st.caption("Model exploration and inference")
    st.divider()
    st.markdown("**Project status**")
    st.write("Model", "Available" if model is not None else "Not found / unavailable")
    st.write("Dataset", dataset_path.name if dataset_path else "Not detected")
    st.divider()
    st.caption("Place `model.pkl` and, optionally, a CSV dataset in the project root.")

st.title("Machine Learning Project")
st.write("Explore the available dataset, inspect the saved estimator, and run predictions on new samples.")
if model is None:
    st.warning("The local model could not be loaded. Add a trusted `model.pkl` file to the project root and restart the app.")
    with st.expander("Model loading details"):
        st.code(model_error or "No model file was found.")
else:
    info = model_summary(model)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model type", info["type"])
    c2.metric("Input features", str(info["features"]) if info["features"] is not None else "Unknown")
    c3.metric("Prediction support", "Classification" if info["classification"] else ("Regression" if info["regression"] else "Estimator loaded"))
if dataset is None:
    st.info("No dataset was detected automatically. Open **Dataset Summary** to upload a CSV for this session.")
elif dataset_error:
    st.warning("A dataset file was detected but could not be read.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset rows", f"{len(dataset):,}")
    c2.metric("Dataset columns", f"{dataset.shape[1]:,}")
    c3.metric("Duplicate rows", f"{int(dataset.duplicated().sum()):,}")
st.markdown("#### Workflow")
st.markdown("1. **Overview** — check which project artifacts are available.  \n2. **Dataset Summary** — review schema, quality, distributions, and filtered records.  \n3. **Test a New Sample** — enter feature values and inspect the model output.")
st.caption("Metrics are computed from local artifacts at runtime. No model performance score is shown unless evaluation results are explicitly provided.")
