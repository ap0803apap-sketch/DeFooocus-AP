import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import modules.config


CHARACTER_PROFILE_DIR = Path('models/character_profiles')
TRAINING_JOB_DIR = Path('models/lora_training_jobs')


@dataclass
class StoryPanel:
    panel: int
    title: str
    prompt: str
    caption: str


@dataclass
class LoraTrainingJob:
    job_name: str
    trigger_word: str
    learning_rate: float
    steps: int
    rank: int
    created_at: str
    reference_images: List[str]
    notes: str


def _safe_slug(text: str, default: str = 'item') -> str:
    if not text:
        return default
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', text.strip()).strip('_').lower()
    return slug or default


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_character_prompt(base_prompt: str, pose: str, clothes: str, environment: str,
                           identity_strength: float, style_strength: float) -> str:
    prompt_parts = [base_prompt.strip()]
    if pose.strip():
        prompt_parts.append(f'pose: {pose.strip()}')
    if clothes.strip():
        prompt_parts.append(f'clothes: {clothes.strip()}')
    if environment.strip():
        prompt_parts.append(f'environment: {environment.strip()}')
    prompt_parts.append(f'preserve facial identity strength {identity_strength:.2f}')
    prompt_parts.append(f'reference style influence {style_strength:.2f}')
    return ', '.join([p for p in prompt_parts if p])


def save_character_profile(profile_name: str, base_prompt: str, negative_prompt: str,
                           reference_images: List[str], notes: str = '') -> Tuple[str, str]:
    _ensure_dir(CHARACTER_PROFILE_DIR)
    profile_slug = _safe_slug(profile_name, default='character_profile')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    profile_root = CHARACTER_PROFILE_DIR / f'{profile_slug}_{timestamp}'
    _ensure_dir(profile_root)

    copied_images = []
    for src in reference_images or []:
        if src and os.path.exists(src):
            dst = profile_root / Path(src).name
            shutil.copy2(src, dst)
            copied_images.append(str(dst))

    manifest = {
        'profile_name': profile_name,
        'base_prompt': base_prompt,
        'negative_prompt': negative_prompt,
        'notes': notes,
        'created_at': timestamp,
        'reference_images': copied_images,
        'guidance': {
            'ip_adapter': 'Use Image Prompt with medium/high weight.',
            'face_embedding': 'Use FaceSwap/ImagePrompt Advanced for identity lock.',
            'lora_auto_training': 'Use exported training package to train a LoRA externally.',
        },
    }

    manifest_path = profile_root / 'profile.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    return f'Saved profile in {profile_root}', json.dumps(manifest, indent=2)


def export_character_profile_zip(profile_json_text: str):
    if not profile_json_text or not profile_json_text.strip():
        return None, 'No profile JSON to export.'

    try:
        profile_data = json.loads(profile_json_text)
    except json.JSONDecodeError:
        return None, 'Invalid profile JSON.'

    profile_name = _safe_slug(profile_data.get('profile_name', 'character_profile'))
    temp_dir = Path(tempfile.mkdtemp(prefix='character_profile_'))
    export_root = temp_dir / profile_name
    _ensure_dir(export_root)

    with open(export_root / 'profile.json', 'w', encoding='utf-8') as f:
        json.dump(profile_data, f, indent=2)

    for img in profile_data.get('reference_images', []):
        if img and os.path.exists(img):
            shutil.copy2(img, export_root / Path(img).name)

    zip_path = temp_dir / f'{profile_name}.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file in export_root.rglob('*'):
            if file.is_file():
                zf.write(file, arcname=str(file.relative_to(export_root.parent)))

    return str(zip_path), f'Exported {zip_path.name}'


def create_lora_training_job(job_name: str, trigger_word: str, learning_rate: float, steps: int, rank: int,
                             reference_images: List[str], notes: str = '') -> Tuple[str, str]:
    _ensure_dir(TRAINING_JOB_DIR)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    job_slug = _safe_slug(job_name, default='lora_job')
    job_root = TRAINING_JOB_DIR / f'{job_slug}_{timestamp}'
    images_dir = job_root / 'images'
    _ensure_dir(images_dir)

    copied_images = []
    for src in reference_images or []:
        if src and os.path.exists(src):
            dst = images_dir / Path(src).name
            shutil.copy2(src, dst)
            copied_images.append(str(dst))

    job = LoraTrainingJob(
        job_name=job_name,
        trigger_word=trigger_word,
        learning_rate=learning_rate,
        steps=int(steps),
        rank=int(rank),
        created_at=timestamp,
        reference_images=copied_images,
        notes=notes,
    )

    with open(job_root / 'training_job.json', 'w', encoding='utf-8') as f:
        json.dump(asdict(job), f, indent=2)

    with open(job_root / 'README.txt', 'w', encoding='utf-8') as f:
        f.write(
            'DeFooocus LoRA Auto-Training Package\n\n'
            'This package stores dataset images and training settings.\n'
            'Run your external trainer (kohya_ss/OneTrainer/etc) with these settings.\n'
            'After training, copy the resulting .safetensors into models/loras and click Model Refresh.\n'
        )

    zip_path = shutil.make_archive(str(job_root), 'zip', root_dir=job_root)
    return f'LoRA training package prepared at {zip_path}', json.dumps(asdict(job), indent=2)


def import_lora_file(uploaded_file: str) -> str:
    if not uploaded_file or not os.path.exists(uploaded_file):
        return 'No LoRA file uploaded.'

    if not uploaded_file.lower().endswith('.safetensors'):
        return 'Please upload a .safetensors file.'

    lora_dir = Path(modules.config.path_loras)
    _ensure_dir(lora_dir)
    dst = lora_dir / Path(uploaded_file).name
    shutil.copy2(uploaded_file, dst)
    return f'Imported LoRA to {dst}. Click Model Refresh to load it.'


def generate_story_panels(story_prompt: str, panel_count: int, add_captions: bool, comic_layout: str):
    panel_count = max(1, min(int(panel_count), 12))
    base = story_prompt.strip()
    if not base:
        return json.dumps({'error': 'Story prompt is empty.'}, indent=2)

    beats = [
        'opening scene, establish setting and atmosphere',
        'character faces first challenge',
        'new clue or discovery changes direction',
        'conflict rises with stakes',
        'major reveal or turning point',
        'resolution and emotional payoff',
    ]

    panels = []
    for idx in range(panel_count):
        beat = beats[idx % len(beats)]
        title = f'Panel {idx + 1}'
        prompt = f'{base}, {beat}, cinematic composition, coherent character identity'
        caption = f'{title}: {beat}.' if add_captions else ''
        panels.append(asdict(StoryPanel(panel=idx + 1, title=title, prompt=prompt, caption=caption)))

    result = {
        'story_prompt': base,
        'comic_layout': comic_layout,
        'panels': panels,
        'tips': {
            'identity': 'Use Character Consistency prompt and Image Prompt for all panels.',
            'continuity': 'Keep seed close or fixed and only change pose/environment phrases.',
        },
    }
    return json.dumps(result, indent=2)


def generate_movie_plan(sequence_notes: str, scene_description: str, fps: int, seconds: int):
    scene_description = scene_description.strip()
    if not scene_description:
        return json.dumps({'error': 'Scene description is empty.'}, indent=2)

    shots = []
    notes = [line.strip() for line in sequence_notes.splitlines() if line.strip()]
    if not notes:
        notes = ['opening shot', 'mid action', 'ending shot']

    for i, note in enumerate(notes, start=1):
        shots.append({
            'shot': i,
            'note': note,
            'prompt': f'{scene_description}, {note}, consistent character and lighting, cinematic frame',
            'duration_seconds': max(1, int(seconds) // max(1, len(notes)))
        })

    movie_plan = {
        'scene_description': scene_description,
        'fps': max(8, min(int(fps), 60)),
        'total_seconds': max(1, int(seconds)),
        'shots': shots,
        'render_hint': 'Generate keyframes per shot, then interpolate with your video backend.',
    }
    return json.dumps(movie_plan, indent=2)
