import shutil
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.m5.alert_phrases import split_ship_number, stretch_number_tail


class NumberTailTests(unittest.TestCase):
    def test_only_split_names_ending_in_four_to_eight_chinese_digits(self):
        self.assertEqual(split_ship_number("鲁威渔一五二四零九，"), ("鲁威渔", "一五二四零九，"))
        self.assertEqual(split_ship_number("浙普渔幺〇二三，"), ("浙普渔", "幺〇二三，"))
        for text in ("海巡一号，", "鲁威渔152409，", "鲁威渔十五万二千四百零九，", "一五二四零九，",
                     "鲁威渔一二三，", "鲁威渔一二三四五六七八九，", "甲船，乙船一二三四，"):
            self.assertIsNone(split_ship_number(text), text)

    def test_original_speed_and_short_audio_are_unchanged(self):
        data = b"\x01\x00" * 100
        with patch("backend.m5.alert_phrases.subprocess.run") as run:
            self.assertEqual(b"".join(stretch_number_tail([data, data], 1, "unused")), data * 2)
            self.assertEqual(b"".join(stretch_number_tail([data], 0.5, "unused")), data)
            run.assert_not_called()

    def test_front_is_streamed_before_processing_tail_and_failure_keeps_original(self):
        data = np.full(24000, 1000, dtype="<i2").tobytes()
        for result in (OSError("unavailable"), SimpleNamespace(stdout=b"bad")):
            with self.subTest(result=result), patch("backend.m5.alert_phrases.subprocess.run") as run:
                if isinstance(result, Exception):
                    run.side_effect = result
                else:
                    run.return_value = result
                stream = stretch_number_tail([data], 0.5, "unused")
                front = next(stream)
                self.assertTrue(data.startswith(front))
                self.assertGreater(len(front), 24000)
                run.assert_not_called()
                self.assertEqual(front + b"".join(stream), data)
                run.assert_called_once()

    @unittest.skipUnless(shutil.which("ffmpeg"), "本地未配置 FFmpeg，镜像构建时执行")
    def test_real_filter_extends_tail_preserves_pitch_and_keeps_front_identical(self):
        samples = (6000 * np.sin(2 * np.pi * 400 * np.arange(48000) / 24000)).astype("<i2")
        pcm = samples.tobytes()
        for speed in (0.5, 0.65):
            with self.subTest(speed=speed):
                output = b"".join(stretch_number_tail([pcm[:48000], pcm[48000:]], speed, shutil.which("ffmpeg")))
                self.assertGreater(len(output), len(pcm))
                self.assertLess(len(output), len(pcm) + 24000)
                self.assertTrue(output.startswith(pcm[:-11760]))
                tail = np.frombuffer(output[-12000:], dtype="<i2").astype(float)
                spectrum = np.abs(np.fft.rfft(tail * np.hanning(len(tail))))
                pitch = np.fft.rfftfreq(len(tail), 1 / 24000)[np.argmax(spectrum)]
                self.assertAlmostEqual(pitch, 400, delta=5)


if __name__ == "__main__":
    unittest.main()
