"""
Doplňkové testy pro benchmark_shared.py.

Pokrývá:
    - BenchmarkSummarizer.export_results_json()  (JSON výstup, struktura, fallback)
    - BenchmarkRunner.run() s více iteracemi a s verify_lossless

Přidat na konec souboru tests/core_tests/test_benchmark_shared.py.
Používají stejné helpery (_make_config, _make_result, _make_metrics,
_make_verification_result) které jsou definovány na začátku toho souboru.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from main import (
    BenchmarkResult,
    CompressionLevel,
    CompressionMetrics,
    CompressorFactory,
    ImageCompressor,
)
from utils.cpu_affinity import IsolationConfig
from utils.verification import VerificationResult
from benchmark_shared import (
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkSummarizer,
)


# Helpery (stejné jako v původním souboru)

def _make_metrics(
    original_size: int = 1_000_000,
    compressed_size: int = 500_000,
    compression_ratio: float = 2.0,
    compression_time: float = 1.0,
    decompression_time: float = 0.5,
    success: bool = True,
    error_message=None,
) -> CompressionMetrics:
    return CompressionMetrics(
        original_size=original_size,
        compressed_size=compressed_size,
        compression_ratio=compression_ratio,
        compression_time=compression_time,
        decompression_time=decompression_time,
        success=success,
        error_message=error_message,
    )


def _make_result(
    image_path=None,
    format_name: str = "TestFmt",
    compression_time: float = 1.0,
    decompression_time: float = 0.5,
    compression_ratio: float = 2.0,
    success: bool = True,
    level_name: str = "BALANCED",
    system_metrics=None,
) -> BenchmarkResult:
    return BenchmarkResult(
        image_path=image_path or Path("img.png"),
        format_name=format_name,
        metrics=_make_metrics(
            compression_time=compression_time,
            decompression_time=decompression_time,
            compression_ratio=compression_ratio,
            success=success,
        ),
        metadata={"compression_level": level_name},
        system_metrics=system_metrics,
    )


def _make_config(tmp_path: Path, **overrides) -> BenchmarkConfig:
    defaults = dict(
        dataset_dir=tmp_path / "dataset",
        output_dir=tmp_path / "output",
        libs_dir=tmp_path / "libs",
        compressor_names=["testcomp"],
        image_paths=[tmp_path / "img.png"],
        compression_levels=[CompressionLevel.BALANCED],
        verify_lossless=False,
        strip_metadata=False,
        num_iterations=1,
        warmup_iterations=0,
        trim_top_n=0,
        monitor_resources=False,
        isolation=IsolationConfig(),
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


def _make_mock_compressor(name="MockComp", extension=".mock"):
    comp = MagicMock(spec=ImageCompressor)
    comp.name = name
    comp.extension = extension
    comp.compress.return_value = _make_metrics()
    return comp


# BenchmarkSummarizer.export_results_json() 

class TestExportResultsJson:
    """Ověřuje strukturu, obsah a chování export_results_json()."""

    def _call_export(self, tmp_path, results=None, verification=None, **cfg_overrides):
        """
        Zavolá export_results_json() s minimální konfigurací a vrátí
        seznam uložených cest a naparsovaná JSON data prvního souboru.
        """
        output_dir = tmp_path / "out"
        cfg = _make_config(tmp_path, output_dir=output_dir, **cfg_overrides)
        results = results or [_make_result(image_path=tmp_path / "img.png")]
        verification = verification or {}

        paths = BenchmarkSummarizer.export_results_json(
            results=results,
            verification_results=verification,
            output_dir=output_dir,
            config=cfg,
        )
        return paths, json.loads(paths[0].read_text()) if paths else (paths, None)

    # základní výstup 

    def test_returns_list_of_paths(self, tmp_path):
        paths, _ = self._call_export(tmp_path)
        assert isinstance(paths, list)
        assert len(paths) >= 1
        assert all(isinstance(p, Path) for p in paths)

    def test_created_file_is_valid_json(self, tmp_path):
        paths, data = self._call_export(tmp_path)
        assert data is not None  # json.loads nehodil výjimku

    def test_output_file_ends_with_json(self, tmp_path):
        paths, _ = self._call_export(tmp_path)
        assert paths[0].suffix == ".json"

    def test_output_file_exists_on_disk(self, tmp_path):
        paths, _ = self._call_export(tmp_path)
        assert paths[0].exists()

    # povinné sekce

    def test_contains_benchmark_info_section(self, tmp_path):
        _, data = self._call_export(tmp_path)
        assert "benchmark_info" in data

    def test_contains_benchmark_config_section(self, tmp_path):
        _, data = self._call_export(tmp_path)
        assert "benchmark_config" in data

    def test_contains_summary_section(self, tmp_path):
        _, data = self._call_export(tmp_path)
        assert "summary" in data

    def test_contains_results_section(self, tmp_path):
        _, data = self._call_export(tmp_path)
        assert "results" in data

    def test_contains_verification_section(self, tmp_path):
        _, data = self._call_export(tmp_path)
        assert "verification" in data

    def test_contains_scenarios_section(self, tmp_path):
        _, data = self._call_export(tmp_path)
        assert "scenarios" in data

    # správné pořadí sekcí

    def test_section_order_matches_code(self, tmp_path):
        """
        Skutečné pořadí klíčů v JSON musí odpovídat pořadí v kódu:
        benchmark_info, benchmark_config, summary, scenarios, results, verification.
        """
        paths, _ = self._call_export(tmp_path)
        data = json.loads(paths[0].read_text())
        keys = list(data.keys())
        expected_order = [
            "benchmark_info",
            "benchmark_config",
            "summary",
            "scenarios",
            "results",
            "verification",
        ]
        assert keys == expected_order

    # obsah benchmark_config
    def test_benchmark_config_embeds_num_iterations(self, tmp_path):
        _, data = self._call_export(tmp_path, num_iterations=3)
        assert data["benchmark_config"]["num_iterations"] == 3

    def test_benchmark_config_embeds_compressor_names(self, tmp_path):
        _, data = self._call_export(tmp_path, compressor_names=["pillow-png"])
        assert "pillow-png" in data["benchmark_config"]["compressors"]

    def test_benchmark_config_embeds_verify_flag(self, tmp_path):
        _, data = self._call_export(tmp_path, verify_lossless=True)
        assert data["benchmark_config"]["verify_lossless"] is True

    def test_benchmark_config_embeds_strip_flag(self, tmp_path):
        _, data = self._call_export(tmp_path, strip_metadata=True)
        assert data["benchmark_config"]["strip_metadata"] is True

    # obsah summary

    def test_summary_total_images_matches_result_count(self, tmp_path):
        results = [
            _make_result(image_path=tmp_path / "a.png"),
            _make_result(image_path=tmp_path / "b.png"),
        ]
        _, data = self._call_export(tmp_path, results=results)
        assert data["summary"]["total_images"] == 2

    def test_summary_successful_count_excludes_failures(self, tmp_path):
        results = [
            _make_result(image_path=tmp_path / "a.png", success=True),
            _make_result(image_path=tmp_path / "b.png", success=False),
        ]
        _, data = self._call_export(tmp_path, results=results)
        assert data["summary"]["successful"] == 1
        assert data["summary"]["failed"] == 1

    # obsah results

    def test_results_list_length_matches_input(self, tmp_path):
        results = [
            _make_result(image_path=tmp_path / "a.png"),
            _make_result(image_path=tmp_path / "b.png"),
        ]
        _, data = self._call_export(tmp_path, results=results)
        assert len(data["results"]) == 2

    def test_results_entry_contains_compression_block(self, tmp_path):
        _, data = self._call_export(tmp_path)
        assert "compression" in data["results"][0]

    def test_results_entry_has_correct_format_name(self, tmp_path):
        results = [_make_result(image_path=tmp_path / "img.png", format_name="OxiPNG")]
        _, data = self._call_export(tmp_path, results=results)
        assert data["results"][0]["format"] == "OxiPNG"

    # soubory per level

    def test_one_file_per_compression_level(self, tmp_path):
        results = [
            _make_result(image_path=tmp_path / "img.png", level_name="FASTEST"),
            _make_result(image_path=tmp_path / "img.png", level_name="BEST"),
        ]
        output_dir = tmp_path / "out"
        cfg = _make_config(
            tmp_path,
            output_dir=output_dir,
            compression_levels=[CompressionLevel.FASTEST, CompressionLevel.BEST],
        )
        paths = BenchmarkSummarizer.export_results_json(
            results=results,
            verification_results={},
            output_dir=output_dir,
            config=cfg,
        )
        assert len(paths) == 2

    # verifikace v JSON 

    def test_verification_data_embedded_in_json(self, tmp_path):
        img_path = tmp_path / "img.png"
        results = [_make_result(image_path=img_path, format_name="Pillow-PNG")]
        verif = {
            ("img.png", "Pillow-PNG"): VerificationResult(
                is_lossless=True,
                max_difference=0.0,
                different_pixels=0,
                total_pixels=16,
            )
        }
        _, data = self._call_export(tmp_path, results=results, verification=verif)
        assert len(data["verification"]) == 1
        assert data["verification"][0]["is_lossless"] is True

    # fallback při selhání zápisu

    def test_fallback_filename_used_when_primary_write_fails(self, tmp_path):
        """
        Pokud zápis na primární cestu selže (simulováno patchem open()),
        export_results_json() musí zapsat na záložní jméno a vrátit jeho cestu.
        """
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True)
        cfg = _make_config(tmp_path, output_dir=output_dir)
        results = [_make_result(image_path=tmp_path / "img.png")]

        call_count = {"n": 0}
        real_open = open

        def failing_open(path, *args, **kwargs):
            # První volání (primární soubor) hodí výjimku
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("disk full")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=failing_open):
            paths = BenchmarkSummarizer.export_results_json(
                results=results,
                verification_results={},
                output_dir=output_dir,
                config=cfg,
            )

        assert len(paths) == 1
        assert paths[0].exists()


# BenchmarkRunner.run(), doplňkové scénáře 

class TestBenchmarkRunnerRunExtra:
    """
    Doplňuje TestBenchmarkRunnerRun z původního souboru o scénáře
    s více iteracemi a s verify_lossless.
    """

    def test_run_with_multiple_iterations_returns_one_averaged_result(self, tmp_path):
        """
        Při num_iterations=3 musí runner vrátit jeden zprůměrovaný výsledek
        na kombinaci kompresor × úroveň × obrázek, ne tři separátní.
        """
        src = tmp_path / "img.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")

        cfg = _make_config(
            tmp_path,
            image_paths=[src],
            compressor_names=["mockcomp"],
            num_iterations=3,
            warmup_iterations=0,
        )
        runner = BenchmarkRunner(cfg)
        mock_comp = _make_mock_compressor()

        with patch.object(CompressorFactory, "create", return_value=mock_comp), \
             patch.object(runner, "_prepare_input", return_value=(src, False)), \
             patch.object(runner, "_find_lib_for_compressor", return_value=None):
            results, _ = runner.run()

        # 1 kompresor × 1 úroveň × 1 obrázek = 1 průměrovaný výsledek
        assert len(results) == 1

    def test_run_calls_compress_num_iterations_times(self, tmp_path):
        """
        compress() musí být zavoláno právě num_iterations-krát
        (zahřívací běhy se nepočítají).
        """
        src = tmp_path / "img.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")

        cfg = _make_config(
            tmp_path,
            image_paths=[src],
            compressor_names=["mockcomp"],
            num_iterations=4,
            warmup_iterations=2,
        )
        runner = BenchmarkRunner(cfg)
        mock_comp = _make_mock_compressor()

        with patch.object(CompressorFactory, "create", return_value=mock_comp), \
             patch.object(runner, "_prepare_input", return_value=(src, False)), \
             patch.object(runner, "_find_lib_for_compressor", return_value=None):
            runner.run()

        # 2 warm-up + 4 měřené = 6 celkem
        assert mock_comp.compress.call_count == 6

    def test_run_with_two_compressors_returns_two_results(self, tmp_path):
        """Každý kompresor musí přispět vlastním výsledkem."""
        src = tmp_path / "img.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")

        cfg = _make_config(
            tmp_path,
            image_paths=[src],
            compressor_names=["comp_a", "comp_b"],
            num_iterations=1,
            warmup_iterations=0,
        )
        runner = BenchmarkRunner(cfg)
        comp_a = _make_mock_compressor(name="CompA", extension=".ca")
        comp_b = _make_mock_compressor(name="CompB", extension=".cb")

        def factory_side_effect(name, lib_path=None):
            return comp_a if name == "comp_a" else comp_b

        with patch.object(CompressorFactory, "create", side_effect=factory_side_effect), \
             patch.object(runner, "_prepare_input", return_value=(src, False)), \
             patch.object(runner, "_find_lib_for_compressor", return_value=None):
            results, _ = runner.run()

        assert len(results) == 2
        names = {r.format_name for r in results}
        assert names == {"CompA", "CompB"}

    def test_run_with_two_images_returns_two_results(self, tmp_path):
        """Každý vstupní obrázek musí mít vlastní výsledek."""
        src_a = tmp_path / "a.png"
        src_b = tmp_path / "b.png"
        Image.new("RGB", (4, 4)).save(src_a, format="PNG")
        Image.new("RGB", (4, 4)).save(src_b, format="PNG")

        cfg = _make_config(
            tmp_path,
            image_paths=[src_a, src_b],
            compressor_names=["mockcomp"],
            num_iterations=1,
            warmup_iterations=0,
        )
        runner = BenchmarkRunner(cfg)
        mock_comp = _make_mock_compressor()

        with patch.object(CompressorFactory, "create", return_value=mock_comp), \
             patch.object(runner, "_prepare_input", return_value=(src_a, False)), \
             patch.object(runner, "_find_lib_for_compressor", return_value=None):
            results, _ = runner.run()

        assert len(results) == 2

    def test_run_with_verify_lossless_populates_verification_dict(self, tmp_path):
        """
        Při verify_lossless=True musí runner vrátit neprázdný slovník
        verifikačních výsledků.
        """
        src = tmp_path / "img.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")

        cfg = _make_config(
            tmp_path,
            image_paths=[src],
            compressor_names=["mockcomp"],
            num_iterations=1,
            warmup_iterations=0,
            verify_lossless=True,
        )
        runner = BenchmarkRunner(cfg)
        mock_comp = _make_mock_compressor()

        fake_verif = VerificationResult(
            is_lossless=True,
            max_difference=0.0,
            different_pixels=0,
            total_pixels=16,
        )

        with patch.object(CompressorFactory, "create", return_value=mock_comp), \
             patch.object(runner, "_prepare_input", return_value=(src, False)), \
             patch.object(runner, "_find_lib_for_compressor", return_value=None), \
             patch.object(runner, "_verify_result", return_value=fake_verif):
            results, verif = runner.run()

        assert len(verif) > 0

    def test_averaged_result_compression_time_is_mean(self, tmp_path):
        """
        Průměrovaný compression_time musí odpovídat aritmetickému průměru
        časů ze všech iterací.
        """
        src = tmp_path / "img.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")

        cfg = _make_config(
            tmp_path,
            image_paths=[src],
            compressor_names=["mockcomp"],
            num_iterations=3,
            warmup_iterations=0,
        )
        runner = BenchmarkRunner(cfg)
        mock_comp = _make_mock_compressor()
        # Tři iterace s různými časy: průměr = (1.0 + 2.0 + 3.0) / 3 = 2.0
        mock_comp.compress.side_effect = [
            _make_metrics(compression_time=1.0),
            _make_metrics(compression_time=2.0),
            _make_metrics(compression_time=3.0),
        ]

        with patch.object(CompressorFactory, "create", return_value=mock_comp), \
             patch.object(runner, "_prepare_input", return_value=(src, False)), \
             patch.object(runner, "_find_lib_for_compressor", return_value=None):
            results, _ = runner.run()

        assert results[0].metrics.compression_time == pytest.approx(2.0)