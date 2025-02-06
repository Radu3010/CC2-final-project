from PIL import Image
import os

# Directorul cu imaginile originale
input_dir = '.venv/landscapes'
# Directorul unde vor fi salvate imaginile procesate
output_dir = '.venv/landscapes1'
os.makedirs(output_dir, exist_ok=True)

# Dimensiunea dorită
target_size = (64, 64)

# Parcurge toate fișierele din directorul de intrare
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        try:
            # Încarcă imaginea
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('RGB')

            # Redimensionează imaginea
            img = img.resize(target_size)

            # Salvează imaginea procesată
            img.save(os.path.join(output_dir, filename))
        except Exception as e:
            print(f"Nu am putut procesa imaginea {filename}. Eroare: {e}")

print('Procesarea imaginilor a fost finalizată. Toate imaginile valide sunt salvate în directorul "landscapes1".')