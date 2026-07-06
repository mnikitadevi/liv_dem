# src/ui.py
"""
Gradio interface for the live demo.

Flow (sequential reveal):
  1. Presenter shows an object (golf ball / chain saw / parachute) via webcam or upload.
  2. Click "Run Model A" -> shows the small scratch-trained model's guess (often shaky).
  3. Click "Reveal Model B" -> shows the pretrained ResNet50's guess on the same image.
  4. A stats table appears comparing both, to visualize the data/compute gap.

Run with:
    uv run main.py
"""

import gradio as gr

from infer import (
    load_model_a,
    load_model_b,
    predict_model_a,
    predict_model_b,
    get_model_a_stats,
    get_model_b_stats,
)

# Load both models once at startup — not per click, so the live demo feels instant.
print("Loading Model A (scratch CNN)...")
MODEL_A, CLASSES_A = load_model_a()
print("Loading Model B (pretrained ResNet50)...")
MODEL_B, CLASSES_B = load_model_b()
print("Both models ready.")


def run_model_a(image):
    if image is None:
        return None, {}, gr.update(visible=False)

    predictions = predict_model_a(image, MODEL_A, CLASSES_A, topk=3)
    label_dict = {label: conf for label, conf in predictions}
    stats = get_model_a_stats(MODEL_A, CLASSES_A)

    # Reveal the "Reveal Model B" button only after Model A has actually run
    return label_dict, stats, gr.update(visible=True)


def run_model_b(image):
    if image is None:
        return None, {}, ""

    predictions = predict_model_b(image, MODEL_B, CLASSES_B, topk=3)
    label_dict = {label: conf for label, conf in predictions}
    stats = get_model_b_stats(MODEL_B, CLASSES_B)

    comparison = (
        "### The gap\n"
        f"Model A saw **{get_model_a_stats(MODEL_A, CLASSES_A)['Training images']} images** "
        f"across **{len(CLASSES_A)} categories**, with no pretraining.\n\n"
        f"Model B saw **~1.28 million images** across **{len(CLASSES_B)} categories** — "
        "trained on serious compute, not a laptop.\n\n"
        "Same kind of task. Wildly different scale. That's the whole story."
    )

    return label_dict, stats, comparison


with gr.Blocks(title="Building an AI, Live") as demo:
    gr.Markdown("# Building an AI — Live")
    gr.Markdown(
        "Show the camera a **golf ball**, a **chain saw**, or a **parachute** — "
        "or upload a photo — and let's see what happens."
    )

    image_input = gr.Image(
        type="pil",
        sources=["upload", "webcam"],
        label="Object to classify",
    )

    gr.Markdown("## Step 1 — Model A: what I trained just now")
    btn_run_a = gr.Button("Run Model A", variant="primary")
    label_a = gr.Label(label="Model A's guess", num_top_classes=3)
    stats_a = gr.JSON(label="Model A stats")

    gr.Markdown("## Step 2 — Model B: what real-world scale looks like")
    btn_reveal_b = gr.Button("Reveal Model B", visible=False, variant="secondary")
    label_b = gr.Label(label="Model B's guess", num_top_classes=3)
    stats_b = gr.JSON(label="Model B stats")
    comparison_md = gr.Markdown()

    btn_run_a.click(
        fn=run_model_a,
        inputs=[image_input],
        outputs=[label_a, stats_a, btn_reveal_b],
    )

    btn_reveal_b.click(
        fn=run_model_b,
        inputs=[image_input],
        outputs=[label_b, stats_b, comparison_md],
    )


def launch():
    demo.launch()


if __name__ == "__main__":
    launch()
