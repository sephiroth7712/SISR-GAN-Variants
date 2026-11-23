# Third-party notices

Parts of this project are based on the open-source projects listed below. Each
adapted file starts with a comment naming its source, and all of them were
modified for this project.

## leftthomas/SRGAN

- Source: https://github.com/leftthomas/SRGAN
- License: MIT (reproduced below)
- Used in: `train.py`, `test.py`, `test_benchmark.py`, `test_video.py`,
  `data.py`, `loss.py`, `models/SRGAN.py`, `models/discriminator.py` and the
  `ResidualBlock` / `UpsampleBLock` modules in `models/common.py`

```
MIT License

Copyright (c) 2020 Hao Ren

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

## sanghyun-son/EDSR-PyTorch

- Source: https://github.com/sanghyun-son/EDSR-PyTorch
- License: MIT (reproduced below)
- Used in: `models/EDSR.py`

```
MIT License

Copyright (c) 2018 Sanghyun Son

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

## Lornatang/ESPCN-PyTorch and Lornatang/FSRCNN-PyTorch

- Sources: https://github.com/Lornatang/ESPCN-PyTorch and
  https://github.com/Lornatang/FSRCNN-PyTorch
- License: Apache License 2.0, the same license as this project; its full text
  is in [LICENSE](LICENSE)
- Copyright notices: "Copyright 2020 Dakewe Biotech Corporation. All Rights
  Reserved." (ESPCN-PyTorch) and "Copyright 2021 Dakewe Biotech Corporation.
  All Rights Reserved." (FSRCNN-PyTorch)
- Used in: `models/ESPCN.py` and `models/FSRCNN.py`, both modified

## yjn870/IDN-pytorch

- Source: https://github.com/yjn870/IDN-pytorch
- License: none specified
- Used in: `models/IDN.py`

The IDN-pytorch repository does not include a license, so it grants no
explicit permission to reuse its code. Check the upstream repository for its
current terms before reusing `models/IDN.py`.
