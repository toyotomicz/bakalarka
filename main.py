import os
import time
import subprocess
from pathlib import Path
from abc import ABC, abstractmethod
from PIL import Image  # Přidáno pro práci s PNG soubory

# Nastavení cest k nástrojům a složkám
BASE_DIR = Path(__file__).resolve().parent  # Získá cestu ke složce, kde se nachází tento skript
CWEBP_PATH = BASE_DIR / "libs" / "cwebp.exe"  # Cesta k nástroji pro kompresi do WebP
DWEBP_PATH = BASE_DIR / "libs" / "dwebp.exe"  # Cesta k nástroji pro dekompresi z WebP

input_path = "input.png"  # Vstupní soubor pro test
output_dir = Path("results")  # Výstupní složka pro výsledky
output_dir.mkdir(exist_ok=True)  # Vytvoří složku, pokud neexistuje

# Abstraktní třída pro všechny kodeky (definuje rozhraní)
class Codec(ABC):
    def __init__(self, name):
        self.name = name  # Název kodeku

    @abstractmethod
    def compress(self, input_file: str, output_file: str, params: str) -> float:
        """Komprese souboru, vrací čas běhu"""
        pass

    @abstractmethod
    def decompress(self, input_file: str, output_file: str) -> float:
        """Dekompresní funkce, vrací čas běhu"""
        pass


# Implementace kodeku pro WebP
class WebPCodec(Codec):
    def __init__(self):
        super().__init__("WebP")

    def compress(self, input_file: str, output_file: str, params: str) -> float:
        # Vytvoří příkaz pro cwebp s danými parametry
        cmd = f'"{CWEBP_PATH}" {params} "{input_file}" -o "{output_file}"'
        start = time.time()
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        end = time.time()
        if result.returncode != 0:
            # Pokud nastane chyba, vyvolá výjimku
            raise RuntimeError(result.stderr.decode())
        return end - start  # Vrací čas komprese

    def decompress(self, input_file: str, output_file: str) -> float:
        # Vytvoří příkaz pro dwebp
        cmd = f'"{DWEBP_PATH}" "{input_file}" -o "{output_file}"'
        start = time.time()
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        end = time.time()
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode())
        return end - start  # Vrací čas dekomprese


# Implementace kodeku pro PNG (pomocí knihovny Pillow)
class PNGCodec(Codec):
    def __init__(self):
        super().__init__("PNG")

    def compress(self, input_file: str, output_file: str, params: str) -> float:
        # Komprese PNG pomocí Pillow
        compression_level = int(params)
        start = time.time()
        image = Image.open(input_file)
        image.save(output_file, format="PNG", compress_level=compression_level)
        end = time.time()
        return end - start  # Vrací čas komprese

    def decompress(self, input_file: str, output_file: str) -> float:
        # Dekompresní fáze u PNG je v podstatě jen znovuuložení souboru
        start = time.time()
        image = Image.open(input_file)
        image.save(output_file, format="PNG")
        end = time.time()
        return end - start  # Vrací čas dekomprese


# Funkce pro testování kodeku s různými parametry
def test_codec(codec: Codec, param_list):
    print(f"\n== Testuje se {codec.name} ==")
    print(f"{'Parametry':<25} | {'Kompresní čas':<15} | {'Dekompresní čas':<17} | {'Velikost (KB)':<15} | {'Poměr'}")
    print("-" * 90)

    for i, params in enumerate(param_list):
        # Nastavení cest pro komprimované a dekomprimované soubory
        ext = "webp" if codec.name == "WebP" else "png"
        compressed_path = output_dir / f"{codec.name}_compressed_{i}.{ext}"
        decompressed_path = output_dir / f"{codec.name}_decompressed_{i}.png"

        try:
            # Spuštění komprese a dekomprese
            t_compress = codec.compress(input_path, compressed_path, params)
            t_decompress = codec.decompress(compressed_path, decompressed_path)

            # Výpočet velikostí a poměru
            input_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(compressed_path)
            ratio = round(input_size / compressed_size, 2) if compressed_size > 0 else 0

            # Výpis výsledků v tabulce
            print(f"{params:<25} | {t_compress:.4f} s       | {t_decompress:.4f} s         | {compressed_size / 1024:.2f} KB     | {ratio}x")

        except RuntimeError as e:
            # Ošetření případné chyby při běhu
            print(f"[!] Chyba u parametru '{params}': {e}")


# Hlavní část programu – spustí testy pro oba kodeky
if __name__ == "__main__":
    webp = WebPCodec()
    webp_params = ["-lossless -z 0", "-lossless -z 3", "-lossless -z 6", "-lossless -z 9"]  # Parametry pro WebP

    png = PNGCodec()
    png_params = ["0", "3", "6", "9"]  # 0 = bez komprese, 9 = maximální komprese

    test_codec(webp, webp_params)
    test_codec(png, png_params)
