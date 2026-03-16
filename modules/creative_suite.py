import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import sys
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


def _to_paths(uploaded) -> List[str]:
    if uploaded is None:
        return []

    candidates = uploaded if isinstance(uploaded, list) else [uploaded]
    paths = []
    for item in candidates:
        if not item:
            continue
        if isinstance(item, str):
            paths.append(item)
            continue

        # gr.File(type='file') usually provides a temp file object with `.name`
        name = getattr(item, 'name', None)
        if isinstance(name, str) and name:
            paths.append(name)
            continue

        # fallback for dict-like structures from some gradio versions
        if isinstance(item, dict):
            maybe_path = item.get('name') or item.get('path')
            if isinstance(maybe_path, str) and maybe_path:
                paths.append(maybe_path)

    return paths


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
    for src in _to_paths(reference_images):
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
    for src in _to_paths(reference_images):
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
    paths = _to_paths(uploaded_file)
    uploaded_file = paths[0] if paths else ''

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


def generate_prompt_variations(subject_prompt: str, style: str, count: int):
    base = (subject_prompt or '').strip()
    if not base:
        return json.dumps({'error': 'Subject prompt is empty.'}, indent=2)

    style = (style or 'cinematic').strip()
    count = max(1, min(int(count), 12))
    camera_setups = [
        'close-up portrait, 85mm lens',
        'medium shot, natural perspective',
        'wide establishing shot, dramatic composition',
        'low-angle hero framing',
        'high-angle atmospheric framing',
        'dynamic motion composition',
    ]
    lighting_setups = [
        'soft diffused lighting',
        'golden hour rim light',
        'volumetric god rays',
        'moody noir contrast',
        'neon bounce lighting',
        'studio key and fill setup',
    ]

    variants = []
    for i in range(count):
        variants.append({
            'id': i + 1,
            'prompt': f'{base}, {style} style, {camera_setups[i % len(camera_setups)]}, '
                      f'{lighting_setups[i % len(lighting_setups)]}, ultra-detailed',
        })

    return json.dumps({'subject': base, 'style': style, 'variations': variants}, indent=2)


def suggest_negative_prompt(subject_prompt: str):
    base = (subject_prompt or '').strip()
    quality_negatives = [
        'lowres', 'blurry', 'out of focus', 'bad anatomy', 'deformed hands', 'extra fingers',
        'asymmetrical eyes', 'jpeg artifacts', 'watermark', 'logo', 'text', 'oversaturated',
    ]

    photo_negatives = ['cartoon', '3d render', 'cgi']
    anime_negatives = ['photorealistic skin pores', 'camera sensor noise']

    base_lower = base.lower()
    if 'anime' in base_lower or 'manga' in base_lower:
        merged = quality_negatives + anime_negatives
    else:
        merged = quality_negatives + photo_negatives

    return ', '.join(merged)


def storyboard_from_single_prompt(master_prompt: str, shot_count: int):
    prompt = (master_prompt or '').strip()
    if not prompt:
        return json.dumps({'error': 'Master prompt is empty.'}, indent=2)

    shot_count = max(1, min(int(shot_count), 10))
    progression = [
        'opening wide shot introducing scene',
        'character-focused medium shot',
        'close-up emotional beat',
        'action or transition shot',
        'detail insert shot',
        'climactic frame',
        'resolution shot',
    ]

    shots = []
    for i in range(shot_count):
        beat = progression[i % len(progression)]
        shots.append({
            'shot': i + 1,
            'beat': beat,
            'prompt': f'{prompt}, {beat}, cinematic continuity, consistent character identity'
        })

    return json.dumps({'master_prompt': prompt, 'shots': shots}, indent=2)


def recommend_generation_preset(creative_goal: str, target_platform: str, speed_priority: str):
    goal = (creative_goal or '').strip().lower()
    platform = (target_platform or 'instagram').strip().lower()
    speed = (speed_priority or 'balanced').strip().lower()

    if not goal:
        return json.dumps({'error': 'Creative goal is empty.'}, indent=2)

    styles = ['Fooocus V2']
    guidance = 5.0
    sharpness = 2.0
    performance = 'Quality'
    negative = suggest_negative_prompt(goal)

    if 'portrait' in goal or 'fashion' in goal or 'face' in goal:
        styles += ['Fooocus Enhance']
        guidance = 5.5
        sharpness = 2.5
    if 'cinematic' in goal or 'movie' in goal:
        styles += ['Fooocus Sharp']
        guidance = 6.0
        sharpness = 3.0
    if 'anime' in goal or 'manga' in goal:
        styles = ['Fooocus V2', 'Fooocus Masterpiece']
        guidance = 4.8
        sharpness = 1.8

    aspect_ratio_map = {
        'instagram': '1024×1024',
        'youtube': '1344×768',
        'tiktok': '768×1344',
        'print': '1152×896',
        'wallpaper': '1536×640',
    }
    aspect_ratio = aspect_ratio_map.get(platform, '1024×1024')

    if speed == 'fast':
        performance = 'Speed'
    elif speed == 'max quality':
        performance = 'Quality'
        guidance += 0.5
        sharpness += 0.3
    else:
        performance = 'Balanced'

    return json.dumps({
        'input': {
            'creative_goal': creative_goal,
            'target_platform': target_platform,
            'speed_priority': speed_priority,
        },
        'recommended': {
            'performance': performance,
            'aspect_ratio': aspect_ratio,
            'styles': styles,
            'guidance_scale': round(guidance, 2),
            'sharpness': round(sharpness, 2),
            'negative_prompt': negative,
        },
        'note': 'Apply these manually in the main generation controls for best results.',
    }, indent=2)


def build_pose_change_plan(subject_image, pose_reference_image, optional_prompt: str = '', keep_face: bool = True):
    subject_paths = _to_paths(subject_image)
    pose_paths = _to_paths(pose_reference_image)
    subject = subject_paths[0] if subject_paths else None
    pose_ref = pose_paths[0] if pose_paths else None

    if not subject or not pose_ref:
        return json.dumps({'error': 'Please upload both subject image and pose reference image.'}, indent=2)

    prompt_parts = [
        'same person as subject image',
        'match exact body pose and camera angle from pose reference image',
        'preserve face identity and skin tone',
    ]
    if optional_prompt and optional_prompt.strip():
        prompt_parts.append(optional_prompt.strip())

    if keep_face:
        prompt_parts.append('high face consistency, do not alter identity')

    return json.dumps({
        'task': 'ai_pose_changer',
        'inputs': {
            'subject_image': subject,
            'pose_reference_image': pose_ref,
            'optional_prompt': optional_prompt,
            'keep_face': keep_face,
        },
        'recommended_prompt': ', '.join(prompt_parts),
        'recommended_negative_prompt': 'different person, identity drift, wrong pose, distorted limbs, extra arms, bad anatomy',
        'how_to_apply': [
            'Go to Input Image -> Inpaint or Outpaint and upload subject image.',
            'Mask full body area to allow pose change.',
            'Go to Image Prompt tab and upload pose reference image in one slot.',
            'Set Image Prompt Type to composition/pose style and medium-high weight.',
            'Paste recommended prompt and generate multiple tries.',
        ],
    }, indent=2)


def build_cloth_change_plan(subject_image, cloth_reference_image, optional_prompt: str = '', preserve_pose: bool = True):
    subject_paths = _to_paths(subject_image)
    cloth_paths = _to_paths(cloth_reference_image)
    subject = subject_paths[0] if subject_paths else None
    cloth_ref = cloth_paths[0] if cloth_paths else None

    if not subject or not cloth_ref:
        return json.dumps({'error': 'Please upload both subject image and cloth reference image.'}, indent=2)

    prompt_parts = [
        'same person and identity as subject image',
        'replace only outfit with clothing from reference image',
        'match texture, fabric pattern, and color exactly',
    ]
    if preserve_pose:
        prompt_parts.append('preserve original pose and body proportions')
    if optional_prompt and optional_prompt.strip():
        prompt_parts.append(optional_prompt.strip())

    return json.dumps({
        'task': 'ai_cloth_changer',
        'inputs': {
            'subject_image': subject,
            'cloth_reference_image': cloth_ref,
            'optional_prompt': optional_prompt,
            'preserve_pose': preserve_pose,
        },
        'recommended_prompt': ', '.join(prompt_parts),
        'recommended_negative_prompt': 'change face, change hairstyle, change pose, wrong clothing colors, mismatched fabric, extra garments',
        'how_to_apply': [
            'Use Inpaint on subject image and mask only clothing region.',
            'Upload cloth reference in Image Prompt tab.',
            'Prefer lower denoise first for fabric transfer, then increase if needed.',
            'Keep face unmasked to preserve identity.',
        ],
    }, indent=2)


def build_expression_change_plan(subject_image, expression_prompt: str = '', expression_reference_image=None,
                                 preserve_identity: bool = True):
    subject_paths = _to_paths(subject_image)
    expr_paths = _to_paths(expression_reference_image)
    subject = subject_paths[0] if subject_paths else None
    expr_ref = expr_paths[0] if expr_paths else None

    if not subject:
        return json.dumps({'error': 'Please upload subject image.'}, indent=2)
    if not expression_prompt.strip() and not expr_ref:
        return json.dumps({'error': 'Provide expression prompt and/or expression reference image.'}, indent=2)

    prompt_parts = ['same person as subject image', 'change only facial expression']
    if expression_prompt.strip():
        prompt_parts.append(f'target expression: {expression_prompt.strip()}')
    if expr_ref:
        prompt_parts.append('match expression muscle pattern from reference image')
    if preserve_identity:
        prompt_parts.append('preserve facial identity, age, skin detail, and head pose')

    return json.dumps({
        'task': 'ai_facial_expression_changer',
        'inputs': {
            'subject_image': subject,
            'expression_prompt': expression_prompt,
            'expression_reference_image': expr_ref,
            'preserve_identity': preserve_identity,
        },
        'recommended_prompt': ', '.join(prompt_parts),
        'recommended_negative_prompt': 'face swap, different identity, asymmetrical eyes, distorted mouth, extra teeth, uncanny face',
        'how_to_apply': [
            'Use Inpaint and mask face region only (eyes, brows, cheeks, mouth).',
            'If available, upload expression reference in Image Prompt tab.',
            'Use moderate denoise to keep identity while changing expression.',
            'Generate multiple candidates and pick best facial anatomy.',
        ],
    }, indent=2)




LOCAL_NSFW_APPS = {
    'nsfw-uncensored-video2': {
        'feature': 'NSFW Uncensored Video2',
        'repo': 'https://huggingface.co/spaces/Heartsync/NSFW-Uncensored-video2',
        'dir': 'NSFW-Uncensored-video2',
        'default_port': 7861,
    },
    'adult-image-generator': {
        'feature': 'Adult Image Generator',
        'repo': 'https://huggingface.co/spaces/Heartsync/Adult',
        'dir': 'Adult',
        'default_port': 7862,
    },
    'nsfw-face-swap': {
        'feature': 'NSFW Face Swap',
        'repo': 'https://huggingface.co/spaces/NRbones/nsfw-face-swap',
        'dir': 'nsfw-face-swap',
        'default_port': 7863,
    },
}

_LOCAL_APP_PROCESSES = {}


def _run_shell(command: str, cwd: Path = None) -> tuple[bool, str]:
    result = subprocess.run(
        command,
        shell=True,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )
    output = (result.stdout or '') + (result.stderr or '')
    return result.returncode == 0, output.strip()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def start_local_nsfw_app(feature_name: str, port: int = 7861) -> str:
    key = (feature_name or '').strip().lower()
    if key not in LOCAL_NSFW_APPS:
        return json.dumps({'error': 'Unknown app key.'}, indent=2)

    app = LOCAL_NSFW_APPS[key]
    root = _project_root()
    app_dir = root / app['dir']

    if key in _LOCAL_APP_PROCESSES:
        proc = _LOCAL_APP_PROCESSES[key]
        if proc.poll() is None:
            return json.dumps({
                'status': 'already_running',
                'feature': app['feature'],
                'port': port,
                'url': f'http://127.0.0.1:{port}',
            }, indent=2)

    if not app_dir.exists():
        ok, out = _run_shell(f"git clone {app['repo']}", cwd=root)
        if not ok:
            return json.dumps({'error': 'Clone failed.', 'details': out}, indent=2)

    venv_dir = app_dir / 'env'
    if not venv_dir.exists():
        ok, out = _run_shell(f'"{sys.executable}" -m venv env', cwd=app_dir)
        if not ok:
            return json.dumps({'error': 'Virtual environment creation failed.', 'details': out}, indent=2)

    python_bin = venv_dir / 'bin' / 'python'
    pip_bin = venv_dir / 'bin' / 'pip'

    req = app_dir / 'requirements.txt'
    if req.exists():
        ok, out = _run_shell(f'"{pip_bin}" install -r requirements.txt', cwd=app_dir)
        if not ok:
            return json.dumps({'error': 'Dependency installation failed.', 'details': out}, indent=2)

    log_file = app_dir / f'run_{port}.log'
    cmd = [
        str(python_bin),
        'app.py',
        '--server_name', '0.0.0.0',
        '--server_port', str(int(port)),
    ]

    with open(log_file, 'a', encoding='utf-8') as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(app_dir),
            stdout=log_f,
            stderr=log_f,
            start_new_session=True,
        )

    _LOCAL_APP_PROCESSES[key] = proc

    return json.dumps({
        'status': 'started',
        'feature': app['feature'],
        'port': int(port),
        'url': f'http://127.0.0.1:{int(port)}',
        'pid': proc.pid,
        'log_file': str(log_file),
        'note': 'Runs inside current runtime (Colab/local machine) and uses local GPU resources.',
    }, indent=2)


def stop_local_nsfw_app(feature_name: str) -> str:
    key = (feature_name or '').strip().lower()
    if key not in LOCAL_NSFW_APPS:
        return json.dumps({'error': 'Unknown app key.'}, indent=2)

    proc = _LOCAL_APP_PROCESSES.get(key)
    if proc is None or proc.poll() is not None:
        return json.dumps({'status': 'not_running', 'feature': LOCAL_NSFW_APPS[key]['feature']}, indent=2)

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

    return json.dumps({'status': 'stopped', 'feature': LOCAL_NSFW_APPS[key]['feature']}, indent=2)


def get_local_nsfw_app_url(feature_name: str, port: int = 7861) -> str:
    key = (feature_name or '').strip().lower()
    if key not in LOCAL_NSFW_APPS:
        return ''
    return f'http://127.0.0.1:{int(port)}'

def external_feature_setup_instructions(feature_name: str):
    key = (feature_name or '').strip().lower()

    recipes = {
        'ai-clothes-changer': {
            'feature': 'AI Clothes Changer',
            'source': 'https://huggingface.co/spaces/frogleo/AI-Clothes-Changer',
            'commands': [
                'git clone https://huggingface.co/spaces/frogleo/AI-Clothes-Changer',
                'cd AI-Clothes-Changer',
                'python -m venv env',
                'source env/bin/activate',
                'pip install -r requirements.txt',
                'python app.py',
            ],
        },
        'outfitanyone': {
            'feature': 'OutfitAnyone',
            'source': 'https://huggingface.co/spaces/HumanAIGC/OutfitAnyone',
            'commands': [
                'git clone https://huggingface.co/spaces/HumanAIGC/OutfitAnyone',
                'cd OutfitAnyone',
                'python -m venv env',
                'source env/bin/activate',
                'pip install -r requirements.txt',
                'python app.py',
            ],
        },
        'expression-editor': {
            'feature': 'Expression Editor',
            'source': 'https://huggingface.co/spaces/fffiloni/expression-editor',
            'commands': [
                "docker run -it -p 7860:7860 --platform=linux/amd64 --gpus all registry.hf.space/fffiloni-expression-editor:latest",
            ],
        },
        'nsfw-uncensored-video2': {
            'feature': 'NSFW Uncensored Video2',
            'source': 'https://huggingface.co/spaces/Heartsync/NSFW-Uncensored-video2',
            'commands': [
                'git clone https://huggingface.co/spaces/Heartsync/NSFW-Uncensored-video2',
                'cd NSFW-Uncensored-video2',
                'python -m venv env',
                'source env/bin/activate',
                'pip install -r requirements.txt',
                'python app.py',
            ],
        },
        'adult-image-generator': {
            'feature': 'Adult Image Generator',
            'source': 'https://huggingface.co/spaces/Heartsync/Adult',
            'commands': [
                'git clone https://huggingface.co/spaces/Heartsync/Adult',
                'cd Adult',
                'python -m venv env',
                'source env/bin/activate',
                'pip install -r requirements.txt',
                'python app.py',
            ],
        },
        'nsfw-face-swap': {
            'feature': 'NSFW Face Swap',
            'source': 'https://huggingface.co/spaces/NRbones/nsfw-face-swap',
            'commands': [
                'git clone https://huggingface.co/spaces/NRbones/nsfw-face-swap',
                'cd nsfw-face-swap',
                'python -m venv env',
                'source env/bin/activate',
                'pip install -r requirements.txt',
                'python app.py',
            ],
        },
    }

    if key not in recipes:
        return json.dumps({
            'error': (
                'Unknown feature. Choose ai-clothes-changer, outfitanyone, expression-editor, '
                'nsfw-uncensored-video2, adult-image-generator, or nsfw-face-swap.'
            )
        }, indent=2)

    recipe = recipes[key]
    return json.dumps({
        'feature': recipe['feature'],
        'source': recipe['source'],
        'commands': recipe['commands'],
        'note': 'These run as external tools. DeFooocus native generation features remain unchanged.',
    }, indent=2)
