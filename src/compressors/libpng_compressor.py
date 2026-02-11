"""
LibPNG Native Compressor Plugin

This module provides a Python interface to the native libpng C library for
high-performance PNG image compression. It uses the `ctypes` foreign function
interface to call libpng functions directly, bypassing slower pure-Python
implementations for the compression step.

Key features of this refactored implementation:
- Cross-platform compatibility for Windows, Linux, and macOS.
- Robust error handling that captures C-level libpng errors and translates
  them into Python exceptions, preventing interpreter crashes.
- Correct and safe memory management when interfacing with NumPy arrays,
  including proper handling of memory layouts and pointer arithmetic.
- A clean separation of concerns between the low-level C library wrapper
  (`LibPNGWriter`) and the high-level compressor plugin interface
  (`LibPNGCompressor`).
"""
# Import standardních knihoven pro práci se C funkcemi
import ctypes
import ctypes.util
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, List, Tuple
from image_size_calculator import ImageSizeCalculator

# Import NumPy pro efektivní práci s obrazovými daty jako s polem
import numpy as np
# pypng je použit pro čtení vstupního PNG souboru a měření času dekompresi,
# poskytuje jednoduchý a bezezávislostní způsob, jak získat pixelová data do NumPy
import png

# Přidání cesty k hlavním modulům projektu umožňuje importy
# z nadřazeného adresáře
sys.path.append(str(Path(__file__).parent.parent))
from main import CompressionLevel, CompressionMetrics, CompressorFactory, ImageCompressor

# --- Type Definitions for C Interoperability ---

# Definice C typů pro prototypy funkcí libpng pomocí ctypes.
# To zajišťuje bezpečnost typů a jasnost při volání C funkcí.
png_byte = ctypes.c_ubyte  # Základní typ pro jeden bajt (0-255)
png_bytep = ctypes.POINTER(png_byte)  # Ukazatel na png_byte (pole bajtů)
png_bytepp = ctypes.POINTER(png_bytep)  # Ukazatel na pole ukazatelů png_bytep
png_structp = ctypes.c_void_p  # Neprůhledný ukazatel na png_struct
png_infop = ctypes.c_void_p  # Neprůhledný ukazatel na png_info
png_voidp = ctypes.c_voidp  # Generický void ukazatel
png_uint_32 = ctypes.c_uint  # Typ pro 32-bitové číslo bez znaménka
png_charp = ctypes.c_char_p  # Ukazatel na řetězec znaků (C string)
png_FILE_p = ctypes.c_void_p  # Reprezentuje neprůhledný FILE* ukazatel z C stdio

# Definice typů ukazatelů funkcí pro callbacky zpracování chyb v libpng.
# Tyto callbacky umožňují volání Python funkcí z C vrstvy
PNG_ERROR_FUNC = ctypes.CFUNCTYPE(None, png_structp, png_charp)  # Callback pro fatální chyby
PNG_WARNING_FUNC = ctypes.CFUNCTYPE(None, png_structp, png_charp)  # Callback pro varování

# --- Constants from libpng.h ---

# Tyto konstanty jsou zrcadleny z libpng hlavičkového souboru pro použití v API voláních
PNG_LIBPNG_VER_STRING = b"1.6.37"  # Konkrétní verze libpng kterou očekáváme
PNG_COLOR_TYPE_RGB = 2  # Konstanta pro barevný typ RGB (3 kanály)
PNG_COLOR_TYPE_RGBA = 6  # Konstanta pro barevný typ RGBA (4 kanály s alfa)
PNG_INTERLACE_NONE = 0  # Bez prokládání obrazu
PNG_COMPRESSION_TYPE_DEFAULT = 0  # Výchozí typ komprese
PNG_FILTER_TYPE_DEFAULT = 0  # Výchozí typ filtru
PNG_FILTER_NONE = 8  # Žádný filtr
PNG_FILTER_SUB = 16  # Subtraktivní filtr
PNG_FILTER_UP = 32  # Filtr nahoru
PNG_FILTER_AVG = 64  # Průměrný filtr
PNG_FILTER_PAETH = 128  # Paeth filtr
PNG_ALL_FILTERS = (
    PNG_FILTER_NONE | PNG_FILTER_SUB | PNG_FILTER_UP | PNG_FILTER_AVG | PNG_FILTER_PAETH
)  # Kombinace všech filtrů pomocí bitové operace OR

# --- Library Loading Utility ---

def _find_libraries(lib_dir: Path) -> Tuple[str, str, str]:
    """
    Dynamicky najde cesty pro libpng, zlib a standardní C knihovnu
    na Windows.

    Args:
        lib_dir: Adresář obsahující propojené libpng a zlib knihovny.

    Returns:
        Tuple obsahující úplné cesty k (libpng, zlib, libc).

    Raises:
        FileNotFoundError: Pokud nelze nalézt požadované knihovny.
    """
    libpng_name = "libpng16.dll"  # Jméno DLL souboru pro libpng na Windows
    zlib_name = "zlib1.dll"  # Jméno DLL souboru pro zlib (kompresní knihovna)
    # Na Windows je 'msvcrt' jméno pro Microsoft C Runtime Library.
    libc_name = "msvcrt"

    libpng_path = lib_dir / libpng_name  # Konstruuji plnou cestu k libpng
    zlib_path = lib_dir / zlib_name  # Konstruuji plnou cestu k zlib

    if not libpng_path.exists():  # Kontroluji zda soubor existuje
        raise FileNotFoundError(f"libpng not found at expected path: {libpng_path}")
    if not zlib_path.exists():  # Kontroluji zda soubor existuje
        raise FileNotFoundError(f"zlib not found at expected path: {zlib_path}")

    # Používám ctypes utility k nalezení standardní C knihovny v systémových cestách
    libc_path = ctypes.util.find_library(libc_name)
    if not libc_path:  # Pokud není knihovna nalezena
        raise FileNotFoundError(f"Standard C library '{libc_name}' not found.")

    # Vracím řetězce (ne Path objekty) protože ctypes je očekává
    return str(libpng_path), str(zlib_path), libc_path


# --- LibPNG Error Handling Callbacks ---

@PNG_ERROR_FUNC  # Dekorátor definuje tuto funkci jako C-callable callback
def _py_png_error_handler(png_ptr: png_structp, message_ptr: png_charp):
    """
    C-callable error handler. Tato funkce je volána libpng když dojde k fatální
    chybě. Vyvolá Python výjimku místo aby libpng volala longjmp,
    což by způsobilo pád Python interpreteru.
    """
    # Konvertuji C řetězec na Python řetězec s tolerancí vůči chybám kódování
    message = ctypes.string_at(message_ptr).decode('utf-8', errors='ignore')
    raise RuntimeError(f"LibPNG fatal error: {message}")

@PNG_WARNING_FUNC  # Dekorátor definuje tuto funkci jako C-callable callback
def _py_png_warning_handler(png_ptr: png_structp, message_ptr: png_charp):
    """
    C-callable warning handler. Tato funkce je volána libpng pro ne-fatální
    problémy. Vyzývá Python varování.
    """
    # Konvertuji C řetězec na Python řetězec s tolerancí vůči chybám kódování
    message = ctypes.string_at(message_ptr).decode('utf-8', errors='ignore')
    warnings.warn(f"LibPNG warning: {message}", RuntimeWarning)


# --- Section 1: Ctypes Wrapper for LibPNG ---

class LibPNGWriter:
    """
    Nízkourovňový wrapper pro volání funkcí z libpng sdílené knihovny.
    Tato třída zpracovává načítání knihovny, definici prototypů funkcí, správu
    zdrojů a zpracování chyb.
    """
    def __init__(self, libpng_path: str, zlib_path: str, libc_path: str):
        """
        Inicializuje wrapper načtením požadovaných sdílených knihoven.
        """
        try:
            # Na Windows přidám adresář knihovny do DLL vyhledávací cesty
            # aby byly závislosti jako zlib nalezeny libpng
            if platform.system() == "Windows":
                import os
                os.add_dll_directory(str(Path(libpng_path).parent))

            # zlib musí být načten jako první protože na něm libpng závisí.
            # Nemusím přímo volat její funkce, ale musí být v paměti
            ctypes.CDLL(zlib_path)
            self.libpng = ctypes.CDLL(libpng_path)  # Načtu hlavní libpng knihovnu
            self.libc = ctypes.CDLL(libc_path)  # Načtu C runtime knihovnu pro I/O
        except OSError as e:  # Pokud se načítání nezdaří
            raise ImportError(f"Failed to load libpng/zlib/libc shared libraries: {e}")

        # Zavolám metodu pro definici prototypů všech C funkcí
        self._define_prototypes()

    def _define_prototypes(self):
        """
        Definuje C prototypy funkcí pro ctypes. To je kritické pro
        správný marshalling argumentů a zajištění že funkce vrací správné typy.
        """
        # png_create_write_struct - vytvoří strukturu pro psaní PNG
        self.libpng.png_create_write_struct.restype = png_structp  # Vrací ukazatel png_struct
        self.libpng.png_create_write_struct.argtypes = [png_charp, png_voidp, png_voidp, png_voidp]  # Argumety

        # png_create_info_struct - vytvoří informační strukturu
        self.libpng.png_create_info_struct.restype = png_infop  # Vrací ukazatel png_info
        self.libpng.png_create_info_struct.argtypes = [png_structp]  # Bere strukturu pro psaní

        # png_init_io - inicializuje I/O s otevřeným souborem
        self.libpng.png_init_io.argtypes = [png_structp, png_FILE_p]  # Struktura a FILE ukazatel

        # png_set_IHDR - nastavuje hlavičku PNG (rozměry, bitová hloubka, barva)
        self.libpng.png_set_IHDR.argtypes = [
            png_structp, png_infop, png_uint_32, png_uint_32, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        # png_set_compression_level - nastavuje úroveň komprese (1-9)
        self.libpng.png_set_compression_level.argtypes = [png_structp, ctypes.c_int]

        # png_set_filter - nastavuje filtrační strategii
        self.libpng.png_set_filter.argtypes = [png_structp, ctypes.c_int, ctypes.c_int]

        # png_write_info - zapíše PNG metadata
        self.libpng.png_write_info.argtypes = [png_structp, png_infop]

        # png_write_image - zapíše pixelová data
        self.libpng.png_write_image.argtypes = [png_structp, png_bytepp]

        # png_write_end - dokončí zápis PNG souboru
        self.libpng.png_write_end.argtypes = [png_structp, png_infop]

        # png_destroy_write_struct - uvolní paměť pro write strukturu a info strukturu
        # Tato funkce očekává ukazatele na ukazatele (png_structpp, png_infopp)
        self.libpng.png_destroy_write_struct.argtypes = [
            ctypes.POINTER(png_structp), ctypes.POINTER(png_infop)
        ]

        # C standardní knihovniny funkce pro I/O s soubory
        self.libc.fopen.restype = png_FILE_p  # Vrací FILE ukazatel
        self.libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]  # Cesta a režim
        self.libc.fclose.argtypes = [png_FILE_p]  # Bere FILE ukazatel

    def write(self, file_path: str, image_data: np.ndarray, compression_level: int, filter_type: int):
        """
        Zapíše NumPy pole do PNG souboru se zadanými parametry komprese.

        Args:
            file_path: Cesta k výstupnímu PNG souboru.
            image_data: NumPy pole tvaru (výška, šířka, kanály) s dtype=np.uint8.
            compression_level: Úroveň zlib komprese (1-9).
            filter_type: PNG filtrační strategie (např. PNG_ALL_FILTERS).
        """
        height, width, channels = image_data.shape  # Rozpakuji rozměry pole

        # Určím barevný typ na základě počtu kanálů
        if channels == 3:
            color_type = PNG_COLOR_TYPE_RGB  # 3 kanály = RGB
        elif channels == 4:
            color_type = PNG_COLOR_TYPE_RGBA  # 4 kanály = RGBA s průhledností
        else:
            raise ValueError("Image data must have 3 (RGB) or 4 (RGBA) channels.")

        # Zajistím že NumPy pole má C-contiguous rozložení v paměti, které
        # libpng očekává když konstruuji ukazatele na řádky
        if not image_data.flags['C_CONTIGUOUS']:
            image_data = np.ascontiguousarray(image_data, dtype=np.uint8)

        fp = None  # Inicializuji FILE ukazatel na None
        png_ptr = None  # Inicializuji png_struct ukazatel na None
        info_ptr = None  # Inicializuji png_info ukazatel na None

        try:
            # Otevřu soubor pomocí C standardní knihovniny funkce
            fp = self.libc.fopen(file_path.encode('utf-8'), b'wb')  # b'wb' = zápis binárně
            if not fp:  # Pokud se otevření nezdařilo
                raise IOError(f"Could not open file for writing: {file_path}")

            # Vytvořím hlavní libpng write strukturu a předám naše Python error handlery
            png_ptr = self.libpng.png_create_write_struct(
                PNG_LIBPNG_VER_STRING, None, _py_png_error_handler, _py_png_warning_handler
            )
            if not png_ptr:  # Pokud vytvoření selhalo
                raise RuntimeError("png_create_write_struct failed.")

            # Vytvořím informační strukturu
            info_ptr = self.libpng.png_create_info_struct(png_ptr)
            if not info_ptr:  # Pokud vytvoření selhalo
                raise RuntimeError("png_create_info_struct failed.")

            # Inicializuji I/O a napíšu PNG hlavičku (IHDR chunk)
            self.libpng.png_init_io(png_ptr, fp)
            self.libpng.png_set_IHDR(
                png_ptr, info_ptr, width, height, 8, color_type,
                PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
            )

            # Nastavím parametry komprese a filtrování
            self.libpng.png_set_compression_level(png_ptr, compression_level)
            self.libpng.png_set_filter(png_ptr, 0, filter_type)

            # Napíšu informační chunks do souboru
            self.libpng.png_write_info(png_ptr, info_ptr)

            # Připravím obrazová data pro libpng. Funkce png_write_image očekává
            # pole ukazatelů, kde každý ukazatel ukazuje na začátek řádku
            row_pointers_type = png_bytep * height  # Typ pro pole height ukazatelů
            row_pointers = row_pointers_type()  # Vytvořím prázdné pole

            base_address = image_data.ctypes.data  # Získám bázi paměť NumPy pole
            row_stride = image_data.strides[0]  # Počet bajtů k přeskočení pro další řádek

            # Pro každý řádek nastavím ukazatel na jeho pozici v paměti
            for i in range(height):
                row_address = base_address + i * row_stride  # Vypočítám adresu řádku
                row_pointers[i] = ctypes.cast(row_address, png_bytep)  # Nastavím ukazatel

            # Napíšu všechna pixelová data
            self.libpng.png_write_image(png_ptr, row_pointers)

            # Napíšu konec PNG souboru
            self.libpng.png_write_end(png_ptr, None)

        finally:
            # Tento blok zajistí že všechny C-level zdroje jsou uvolněny,
            # i když během zápisu dojde k chybě
            if png_ptr or info_ptr:  # Pokud byla nějaká struktura vytvořena
                # Musím předat ukazatele na naše ukazatelské proměnné (png_structpp, png_infopp)
                # aby je C funkce mohla anulovat po uvolnění
                png_ptr_ref = ctypes.c_void_p(png_ptr)
                info_ptr_ref = ctypes.c_void_p(info_ptr)
                self.libpng.png_destroy_write_struct(
                    ctypes.byref(png_ptr_ref), ctypes.byref(info_ptr_ref)
                )
            if fp:  # Pokud je soubor otevřen
                self.libc.fclose(fp)  # Zavřu soubor


# --- Section 2: Implementation of the Compressor Plugin ---

class LibPNGCompressor(ImageCompressor):
    """
    Vysokourovňový PNG kompresor který využívá nativní libpng knihovnu
    prostřednictvím ctypes wrapperu `LibPNGWriter`.
    """
    def __init__(self, lib_path: Optional[Path] = None):
        self.libpng_writer: Optional = None  # Inicializuji na None, bude nastaveno v _validate_dependencies
        # Volání `super().__init__` spustí `_validate_dependencies`
        super().__init__(lib_path)

    def _validate_dependencies(self) -> None:
        """
        Ověřuje existenci požadovaných libpng a zlib sdílených knihoven
        a inicializuje LibPNGWriter.
        """
        # Určím adresář kde jsou uloženy propojené knihovny
        base_dir = Path(__file__).parent.parent  # Nadřazený adresář tohoto souboru
        libpng_dir = base_dir / "libs" / "libpng"  # Cesta k adresáři s libpng

        if not libpng_dir.is_dir():  # Kontroluji zda adresář existuje
            raise RuntimeError(f"Directory with libpng libraries not found: {libpng_dir}")

        # Najdu cesty knihoven specifické pro platformu
        libpng_path, zlib_path, libc_path = _find_libraries(libpng_dir)

        # Inicializuji nízkourovňový writer s nalezenými cestami
        self.libpng_writer = LibPNGWriter(libpng_path, zlib_path, libc_path)

    @property  # Toto je property, volá se jako atribut ne jako metoda
    def name(self) -> str:
        """Vrací název tohoto kompresoru"""
        return "LibPNG-Native"

    @property  # Toto je property
    def extension(self) -> str:
        """Vrací příponu souboru kterou tento kompresor produkuje"""
        return ".png"

    def compress(
        self,
        input_path: Path,
        output_path: Path,
        level: CompressionLevel = CompressionLevel.BALANCED
    ) -> CompressionMetrics:
        """
        Komprimuje obrázek do PNG formátu pomocí nativní libpng knihovny.
        """
        if not self.libpng_writer:  # Kontroluji zda je writer inicializován
            # Tato kontrola zajistí že byl kompresor správně inicializován
            raise RuntimeError(
                "LibPNG writer is not initialized. Dependency validation may have failed."
            )

        try:
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)  # Získám původní velikost souboru
            start_time = time.perf_counter()  # Zaznamenám čas začátku

            # Načtu vstupní obrázek do NumPy pole pomocí pypng
            reader = png.Reader(filename=str(input_path))
            width, height, pixels, meta = reader.read()  # Čtu PNG metadata a pixely

            # Konvertuji iterátor pixelů do NumPy pole
            pixel_array = np.vstack(list(map(np.uint8, pixels)))
            channels = meta.get('planes', 3)  # Počet barevných kanálů (výchozí 3)
            # Přetvářím pole do tvaru (výška, šířka, kanály)
            image_data = pixel_array.reshape(height, width, channels)

            # Mapuji abstraktní úroveň komprese na specifické libpng parametry
            # pro úroveň komprese a filtrační strategii
            level_map = {
                CompressionLevel.FASTEST: (1, PNG_FILTER_NONE),  # Nejrychlejší = minimální filtrování
                CompressionLevel.BALANCED: (6, PNG_ALL_FILTERS),  # Vyvážená = všechny filtry
                CompressionLevel.BEST: (9, PNG_FILTER_PAETH),  # Nejlepší = nejlepší Paeth filtr
            }
            # Získám úroveň komprese a typ filtru, nebo použiji výchozí
            compression_level, filter_type = level_map.get(level, level_map[CompressionLevel.BALANCED])

            # Provedu kompresi pomocí nativního writeru
            self.libpng_writer.write(
                str(output_path),
                image_data,
                compression_level=compression_level,
                filter_type=filter_type
            )

            compression_time = time.perf_counter() - start_time  # Vypočítám čas komprese
            compressed_size = output_path.stat().st_size  # Získám komprimovanou velikost

            # Měřím čas potřebný na dekompresi výsledného souboru
            decompression_time = self.decompress(output_path, None)

            # Vracím metriky komprese
            return CompressionMetrics(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=original_size / compressed_size if compressed_size > 0 else float('inf'),
                compression_time=compression_time,
                decompression_time=decompression_time,
                success=True
            )

        except Exception as e:  # Chytám jakoukoliv výjimku
            # (z file I/O, pypng nebo našeho libpng wrapperu)
            # a hlásím ji jako selhání metriky
            return CompressionMetrics(
                original_size=0,
                compressed_size=0,
                compression_ratio=0,
                compression_time=0,
                decompression_time=0,
                success=False,
                error_message=str(e)
            )

    def decompress(self, input_path: Path, output_path: Optional[Path]) -> float:
        """
        Měří čas dekomprese čtením PNG souboru. Parametr `output_path`
        je ignorován protože tato metoda je pouze pro měření času.
        """
        start_time = time.perf_counter()  # Zaznamenám čas začátku
        # Používám pypng pro spolehlivý a jednoduchý způsob čtení PNG dat,
        # což slouží jako proxy pro měření výkonu dekomprese
        reader = png.Reader(filename=str(input_path))
        reader.read()  # Pouze čtu data, neukládám je
        return time.perf_counter() - start_time  # Vracím uplynulý čas


# Registruji tento kompresor ve factory aby mohl být dynamicky instancován
CompressorFactory.register("libpng", LibPNGCompressor)