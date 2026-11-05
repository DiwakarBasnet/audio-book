import json
from pathlib import Path
import tempfile
import unittest
import wave

from audio_book.pipeline import job_lock, prepare, render
from audio_book.text import chunk_text, normalize


class FakeEngine:
    identity = 'test-voice-v1'

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def synthesize(self, text, target):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('Simulated interruption')
        with wave.open(str(target), 'wb') as stream:
            stream.setparams((1, 2, 22050, 0, 'NONE', 'not compressed'))
            stream.writeframes(b'\x00\x00' * 100)


class PipelineTests(unittest.TestCase):
    def test_concurrent_writer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with job_lock(Path(directory)):
                with self.assertRaisesRegex(ValueError, 'Another process'):
                    with job_lock(Path(directory)):
                        self.fail('Second writer acquired lock')

    def test_chunking_preserves_content_and_bound(self):
        text = normalize('First sentence.\nWrapped line.\n\n' + 'word ' * 100 + 'x' * 250)
        chunks = chunk_text(text, 100)
        self.assertTrue(all(0 < len(chunk) <= 100 for chunk in chunks))
        self.assertEqual(''.join(text.split()), ''.join(''.join(chunks).split()))

    def test_interrupted_job_resumes_and_repairs_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'book.txt'
            source.write_text('One paragraph.\n\nSecond paragraph.\n\nThird paragraph.')
            job = Path(directory) / 'job'
            prepare(source, job, 100)
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                render(job, FakeEngine(fail_at=2))
            checkpoint = json.loads((job / 'manifest.json').read_text())
            self.assertEqual(len(checkpoint['completed']), 1)
            resumed = FakeEngine()
            target = render(job, resumed)
            self.assertEqual(resumed.calls, 2)
            with wave.open(str(target), 'rb') as stream:
                self.assertEqual(stream.getnframes(), 300)
            (job / 'chunks/000001.wav').write_bytes(b'corrupt')
            repaired = FakeEngine()
            render(job, repaired)
            self.assertEqual(repaired.calls, 1)
            cached = FakeEngine()
            render(job, cached)
            self.assertEqual(cached.calls, 0)

    def test_sample_does_not_replace_full_output_and_voice_is_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'book.txt'
            source.write_text('One.\n\nTwo.')
            job = Path(directory) / 'job'
            prepare(source, job, 100)
            engine = FakeEngine()
            self.assertEqual(render(job, engine, limit=1).name, 'sample.wav')
            self.assertFalse((job / 'audiobook.wav').exists())
            render(job, engine)
            self.assertEqual(engine.calls, 2)
            different = FakeEngine()
            different.identity = 'different-voice'
            with self.assertRaisesRegex(ValueError, 'Voice or speed'):
                render(job, different)
            with self.assertRaisesRegex(ValueError, 'already contains'):
                prepare(source, job, 100)

    def test_new_checkpoints_bind_audio_to_text(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'book.txt'
            source.write_text('Original paragraph.')
            job = Path(directory) / 'job'
            prepare(source, job, 100)
            render(job, FakeEngine())
            manifest_path = job / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['chunks'][0] = 'Changed narration text.'
            manifest_path.write_text(json.dumps(manifest))
            engine = FakeEngine()
            render(job, engine)
            self.assertEqual(engine.calls, 1)


if __name__ == '__main__':
    unittest.main()
