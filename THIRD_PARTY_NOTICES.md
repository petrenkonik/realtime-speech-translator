# Third-party notices

Model files are downloaded by `scripts/fetch_models.py` into `models/` (never committed)
and are pinned by URL and sha256. Their licenses are reproduced here.

## Silero VAD (`models/silero_vad.onnx`)

Source: <https://github.com/snakers4/silero-vad>, tag `v6.2.3`.

```text
MIT License

Copyright (c) 2020-present Silero Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Smart Turn v3.2 (`models/smart-turn/smart-turn-v3.2-gpu.onnx` and `-cpu.onnx`)

Source: <https://huggingface.co/pipecat-ai/smart-turn-v3>, revision
`f766f81d3cfdf7737ac64aad813d91bbfd56bf93` (model card license: `bsd-2-clause`).
The feature pipeline in `src/pipeline/turn_features.py` is an independent NumPy
implementation that follows the vendor's published pre-processing.

```text
BSD 2-Clause License

Copyright (c) 2024-2025, Daily

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Whisper base.en and tiny.en (`models/whisper-base.en/`, `models/whisper-tiny.en/`)

Weights: OpenAI Whisper `base.en` and `tiny.en` (<https://github.com/openai/whisper>),
converted to the CTranslate2 format by Systran,
<https://huggingface.co/Systran/faster-whisper-base.en>, revision
`3d3d5dee26484f91867d81cb899cfcf72b96be6c`, and
<https://huggingface.co/Systran/faster-whisper-tiny.en>, revision
`0d3d19a32d3338f10357c0889762bd8d64bbdeba` (both model cards: license `mit`). Runs with
`faster-whisper` (MIT, Systran) on `ctranslate2` (MIT, OpenNMT) and `tokenizers` (Apache-2.0).

```text
MIT License

Copyright (c) 2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## OPUS-MT English to German and French (`models/opus-mt-en-de/`, `models/opus-mt-en-fr/`)

Translation models by the Language Technology Research Group at the University of Helsinki
(Helsinki-NLP), trained on OPUS data, <https://github.com/Helsinki-NLP/Opus-MT>:

- `opus-mt-en-de`, <https://huggingface.co/Helsinki-NLP/opus-mt-en-de>, license **CC BY 4.0**
  (<https://creativecommons.org/licenses/by/4.0/>). Attribution is the only condition: the
  credit above and this notice. **Changes made:** converted to the CTranslate2 format
  (float16) by michaelfeil, <https://huggingface.co/michaelfeil/ct2fast-opus-mt-en-de>,
  revision `8ccfcb72178a843301fab5a288dee687e0aa1afc`, and quantized to int8 when loaded.
- `opus-mt-en-fr`, <https://huggingface.co/Helsinki-NLP/opus-mt-en-fr>, license **Apache-2.0**
  (text below), converted the same way,
  <https://huggingface.co/michaelfeil/ct2fast-opus-mt-en-fr>, revision
  `55bcef18761227161433d86b81bbbcd9d813b429`.

They run on `ctranslate2` (MIT, OpenNMT) with `sentencepiece` (Apache-2.0, Google,
<https://github.com/google/sentencepiece>) for tokenization. `sentencepiece` publishes no license
field in its wheel metadata, so `scripts/check_licenses.py` decides it by name.

```text

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## Runtime libraries that need a remark

- **tqdm** (`MPL-2.0 AND MIT`), pulled in by `faster-whisper` and `huggingface-hub`. MPL-2.0
  applies per file to anyone who modifies tqdm's own files; it is imported unmodified, so no
  obligation reaches this project's code. `scripts/check_licenses.py` accepts it by name and
  reports it as a warning.
- **Tk (`tkinter`)** drives the control window (`python -B -m pipeline gui`). It is part of the
  Python standard library (Python Software Foundation License) and the Windows Python installer
  ships Tcl/Tk under its BSD-style license (<https://www.tcl-lang.org/software/tcltk/license.html>).
  It is not listed in `pyproject.toml`, so `scripts/check_licenses.py` does not see it; it uses no
  network and writes no files.
- **PyAV (`av`)** is deliberately not installed. It declares BSD-3-Clause, but its wheels
  bundle an FFmpeg build with LGPL libraries (libmp3lame, opencore-amr). `faster-whisper`
  imports it only to decode audio files, which this project never does, so `pyproject.toml`
  removes it from the resolution and `pipeline.transcribe.ensure_av_stub()` supplies a stub.
  `scripts/check_licenses.py` fails if `av` is ever installed.
- **Windows speech voices (SAPI 5)** speak the translations. They are an operating system
  component that the user installs with a Windows language pack (for example Microsoft Hedda
  Desktop for German and Microsoft Hortense Desktop for French); this project neither bundles
  nor downloads them, and the voices stay under Microsoft's license for Windows. They are
  reached through **comtypes** 1.4.17 (MIT, <https://github.com/enthought/comtypes>), and the
  speech goes into an in-memory stream, never to a file.

## Evaluation data and tools (not shipped)

These are used only by `scripts/eval_endpointing.py` (offline measurement of phrase-end
detection, `docs/SPEC-LOW-LATENCY.md` section 10). They are not part of the demo, are not
imported by `src/`, and are never committed (`data/` is in `.gitignore`).

- **LibriSpeech** `test-clean` audio, `data/librispeech/`. Source:
  <https://www.openslr.org/12>. Paper: V. Panayotov et al., "Librispeech: an ASR corpus based
  on public domain audio books", ICASSP 2015. License: Creative Commons Attribution 4.0
  International (CC BY 4.0), <https://creativecommons.org/licenses/by/4.0/>. Changes made:
  utterances are decoded, trimmed to speech and joined into streams with silences of known
  length. The license is not in the permissive list above because it applies to test data
  only; it is not a dependency and not part of any shipped file.
- **LibriSpeech-PC** manifests (restored punctuation and capitalization), `data/librispeech-pc/`.
  Source: <https://www.openslr.org/145/>, NVIDIA, CC BY 4.0. Paper: A. Meister et al.,
  "LibriSpeech-PC: Benchmark for Evaluation of Punctuation and Capitalization Capabilities of
  end-to-end ASR Models", arXiv:2310.02943 (2023). The sentence-end labels come from its
  `text_raw` field.
- **miniaudio** 1.72 (`miniaudio==1.72`, dev extra), FLAC decoding. License: MIT.
