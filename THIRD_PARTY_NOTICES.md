# Third-party components

Snap Vanish bundles the following components, unmodified except for removed source-map comments.

| Component | Files | Licence |
| --- | --- | --- |
| MediaPipe Tasks Vision 1.1.1 (Google) | `vendor/mediapipe/` | Apache License 2.0 |
| MediaPipe Hand Landmarker model (float16) | `models/hand_landmarker.task` | Apache License 2.0 |
| MediaPipe Selfie Segmenter (landscape) model | `models/selfie_segmenter_landscape.tflite` | Apache License 2.0 |
| ONNX Runtime Web 1.30.0 (Microsoft) | `vendor/ort/` | MIT License |
| MI-GAN inpainting pipeline v2 (Picsart AI Research), ONNX export | `models/migan_pipeline_v2.onnx` | MIT License |

- MediaPipe: https://github.com/google-ai-edge/mediapipe
- ONNX Runtime: https://github.com/microsoft/onnxruntime
- MI-GAN: https://github.com/Picsart-AI-Research/MI-GAN — model file from https://huggingface.co/andraniksargsyan/migan
