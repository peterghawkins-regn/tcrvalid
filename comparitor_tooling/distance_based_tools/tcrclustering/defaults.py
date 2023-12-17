# Copyright 2023 Regeneron Pharmaceuticals Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

DEFAULT_CONDA_PREFIX = '/data/miniconda3'

DEFAULT_ENV = 'tcrvalid'

_path = os.path.dirname(os.path.realpath(__file__))
_ismart_path = os.path.join(_path,'../scripts/modified_third_party/ismart/iSMARTv3.py')
DEFAULT_ISMART_PATH = _ismart_path
DEFAULT_ISMART_env = 'ismart'

