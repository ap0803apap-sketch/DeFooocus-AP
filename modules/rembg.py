import gradio as gr
from PIL import Image


def rembg_run(path, progress=gr.Progress(track_tqdm=True)):
    input_image = Image.open(path)

    try:
        from rembg import remove
    except Exception as e:
        print(f'[Warning] rembg is unavailable and background removal was skipped: {e}')
        return input_image

    return remove(input_image)
