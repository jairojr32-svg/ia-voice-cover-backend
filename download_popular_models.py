import os
import urllib.request
import ssl
import zipfile
import tempfile
import shutil

ssl._create_default_https_context = ssl._create_unverified_context

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

# Highly active public Hugging Face zip downloads (from QuickWick/Music-AI-Voices)
VOICES = {
    "SpongeBob": {
        "url": "https://huggingface.co/airomix/AICG_Models/resolve/main/%23%20SpongeBob%20SquarePants/model.pth",
        "is_zip": False
    },
    "Kanye West": {
        "url": "https://huggingface.co/QuickWick/Music-AI-Voices/resolve/main/Kanye%20West%20(RVC)%201000%20Epoch/Kanye%20West%20(RVC)%201000%20Epoch.zip",
        "is_zip": True
    },
    "Hatsune Miku": {
        "url": "https://huggingface.co/QuickWick/Music-AI-Voices/resolve/main/Hatsune%20Miku%20V2%20-%20VOCALOID%20(RVC)%20250%20Epoch/Hatsune%20Miku%20V2%20-%20VOCALOID%20(RVC)%20250%20Epoch.zip",
        "is_zip": True
    },
    "Ariana Grande": {
        "url": "https://huggingface.co/QuickWick/Music-AI-Voices/resolve/main/Ariana%20Grande%20(RVC)%204k%20Epoch%2028k%20Steps/Ariana%20Grande%20(RVC)%204k%20Epoch%2028k%20Steps.zip",
        "is_zip": True
    },
    "Drake": {
        "url": "https://huggingface.co/binant/Drake_RVC/resolve/main/model.pth",
        "is_zip": False,
        "index_url": "https://huggingface.co/binant/Drake_RVC/resolve/main/model.index"
    },
}

def dl_file(url, path):
    """Downloads a file to a path with a nice progress logger."""
    print(f"Downloading {url} -> {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    )
    with urllib.request.urlopen(req) as response, open(path, "wb") as out_file:
        length = response.getheader('content-length')
        total_size = int(length) if length else None
        downloaded = 0
        block_size = 1024 * 64
        
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            out_file.write(buffer)

def clean_and_extract_zip(zip_path, target_dir, voice_name):
    """Extracts a zip file and reorganizes .pth and .index files to the root."""
    print(f"Extracting zip to {target_dir}...")
    temp_extract_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
            
        # Scan for pth and index files inside the extracted tree
        pth_file = None
        index_file = None
        
        for root, dirs, files in os.walk(temp_extract_dir):
            for file in files:
                if file.endswith(".pth") and not file.startswith("._"):
                    pth_file = os.path.join(root, file)
                elif file.endswith(".index") and not file.startswith("._"):
                    index_file = os.path.join(root, file)
                    
        # Create standard RVC model folder
        os.makedirs(target_dir, exist_ok=True)
        
        if pth_file:
            target_pth = os.path.join(target_dir, f"{voice_name}.pth")
            shutil.move(pth_file, target_pth)
            print(f"✓ Reorganized .pth to: {target_pth}")
        else:
            print(f"✗ Warning: No .pth file found in zip for {voice_name}")
            
        if index_file:
            target_index = os.path.join(target_dir, f"{voice_name}.index")
            shutil.move(index_file, target_index)
            print(f"✓ Reorganized .index to: {target_index}")
            
    finally:
        shutil.rmtree(temp_extract_dir, ignore_errors=True)

def download_all():
    print("--- Starting AI Voice Cover Models Downloader ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for name, config in VOICES.items():
        print(f"\nProcessing voice: {name}")
        voice_folder = os.path.join(MODELS_DIR, name)
        pth_path = os.path.join(voice_folder, f"{name}.pth")
        
        # Skip if already fully downloaded
        if os.path.exists(pth_path):
            print(f"✓ {name} is already available.")
            continue
            
        if config["is_zip"]:
            # Download zip and extract
            temp_zip = os.path.join(tempfile.gettempdir(), f"{name}_model.zip")
            try:
                dl_file(config["url"], temp_zip)
                clean_and_extract_zip(temp_zip, voice_folder, name)
                print(f"✓ Successfully set up voice: {name}")
            except Exception as e:
                print(f"✗ Failed to download/extract zip for {name}: {e}")
            finally:
                if os.path.exists(temp_zip):
                    os.remove(temp_zip)
        else:
            # Download raw .pth file directly
            try:
                dl_file(config["url"], pth_path)
                print(f"✓ Downloaded .pth for {name}")
            except Exception as e:
                print(f"✗ Error downloading pth for {name}: {e}")
                
            # Download raw .index file if available
            if "index_url" in config and config["index_url"]:
                index_path = os.path.join(voice_folder, f"{name}.index")
                if not os.path.exists(index_path):
                    try:
                        dl_file(config["index_url"], index_path)
                        print(f"✓ Downloaded .index for {name}")
                    except Exception as e:
                        print(f"✗ Error downloading index for {name}: {e}")

    print("\nAll default voice models processed!")

if __name__ == "__main__":
    download_all()