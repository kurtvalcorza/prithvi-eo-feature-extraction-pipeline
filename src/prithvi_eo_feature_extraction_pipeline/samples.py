"""Labelled-chip dataset contract for adapting the burn-scar model: the pinned HLS Burn Scars sample, role
assignment from the model repository's splits, BYOD loaders and sample export.

The default dataset is **real**: 44 labelled 512 × 512 HLS scenes of the HLS Burn Scars dataset (NASA IMPACT /
UAH, CC BY 4.0) — 24 from the training split, 8 from the validation split and 12 from the test split that the
upstream authors published beside the checkpoint (non-overlapping; the test split is what their reported IoU was
measured on), drawn with a fixed seed on 2026-09-19 from the scenes whose mask is at least 60 % valid and at least
3 % burn scar, so every chip can be scored. The dataset is distributed as one 2.6 GB gzipped tarball on the Hugging
Face Hub; the tarball is pinned by byte size and SHA-256, each pinned member is pinned again by size and SHA-256
and extracted **without** `extractall` into the cache, and everything else in the archive is left alone. The
repository redistributes none of the scenes.

A record is ``{id, image, label}``: a (6, 512, 512) reflectance array (or a GeoTIFF path) and a (512, 512) mask
with 0 = not burned, 1 = burn scar, -1 = no data (or a GeoTIFF path).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import (
    IMAGE_SIZE,
    MIN_RECORDS,
    MODEL_ID,
    check_record,
    chip_digest,
    read_chip,
    read_mask,
    validate_dataset,
)

CORPUS_NAME = "HLS Burn Scars scenes (NASA IMPACT / UAH)"
CORPUS_RELEASE = (
    "Hugging Face dataset ibm-nasa-geospatial/hls_burn_scars, 44 scenes selected 2026-09-19 from the model repository's splits"
)
CORPUS_LICENSE = "CC BY 4.0 (NASA IMPACT / University of Alabama in Huntsville)"
DATASET_ID = "ibm-nasa-geospatial/hls_burn_scars"
DATASET_REVISION = "1864285e25010d346a842e4f068b1a1d4248ed6d"
CORPUS_BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/"
TAR_NAME = "hls_burn_scars.tar.gz"
TAR_BYTES = 2_645_552_531
TAR_SHA256 = "4e6f99a75cb2c500547b20662a15cbd531dc421376f815e91846ea542798e8e6"
CORPUS_BYTES = 300_099_360  # the 88 pinned members, uncompressed
DEFAULT_CACHE_DIR = Path("weights") / "hls-burn-scars"
ROLES = ("train", "validation", "test")
# (scene key, role from the model repository's splits, image member, image bytes, image sha256, mask member,
#  mask bytes, mask sha256) — member paths are relative to the tarball root
SAMPLE_RECORDS: tuple[tuple[str, str, str, int, str, str, int, str], ...] = (
    (
        "T10SFE.2020267.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T10SFE.2020267.v1.4_merged.tif",
        6295168,
        "98e30b354bbceed46d3e6e1d2c3178c6fba27bf13f9a231ad12175c2374de16b",
        "training/subsetted_512x512_HLS.S30.T10SFE.2020267.v1.4.mask.tif",
        525272,
        "2b615ce4738332d4ccce0aa30a45254c828b28c2f68f2b73a395394a6d813eff",
    ),
    (
        "T10SGE.2019187.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T10SGE.2019187.v1.4_merged.tif",
        6295168,
        "d7fe88a2751e33808afc5fe07732e87c6028d09be3cceda5fe6eb17a4061feda",
        "training/subsetted_512x512_HLS.S30.T10SGE.2019187.v1.4.mask.tif",
        525272,
        "71596c6a730fe28e430ce457ee293039e7311ffa3f1db3d99438dcae754917cc",
    ),
    (
        "T10SGE.2020162.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T10SGE.2020162.v1.4_merged.tif",
        6295168,
        "a6fc23eeb0c070f223ad2eee8a218eab43c74f62d88316e3d0f3db48803c9f37",
        "training/subsetted_512x512_HLS.S30.T10SGE.2020162.v1.4.mask.tif",
        525272,
        "447d5416dc961f5d49a99a07ec44ca3f5149ea07f95604898d214a526a252994",
    ),
    (
        "T10SGG.2020247.v1",
        "train",
        "validation/subsetted_512x512_HLS.S30.T10SGG.2020247.v1.4_merged.tif",
        6295168,
        "51baf73bf405eefff177adc082fcfedbede30f50f92abdf8458880aa89aabeee",
        "validation/subsetted_512x512_HLS.S30.T10SGG.2020247.v1.4.mask.tif",
        525272,
        "2e661b9a46b70c275f0ae3e22594031c56c8c463b288e941159cf628872552c2",
    ),
    (
        "T10TFN.2018245.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T10TFN.2018245.v1.4_merged.tif",
        6295168,
        "5b8ca4c160402252d1c6f373ebee7ea9e197337bf6b768b20d6f10b266d64f49",
        "training/subsetted_512x512_HLS.S30.T10TFN.2018245.v1.4.mask.tif",
        525272,
        "668ce579477b7dc5c7dc537a4bd44ff6cc329d2f31b084f005f9cb7432efa1f2",
    ),
    (
        "T10TFQ.2019245.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T10TFQ.2019245.v1.4_merged.tif",
        6295168,
        "3b4f9f73d8b0b952bca710b96e4cac6d543baa95fe9dc7ee54c7479702e33704",
        "training/subsetted_512x512_HLS.S30.T10TFQ.2019245.v1.4.mask.tif",
        525272,
        "60c898f6f9ecc5710c934ebb5f576a96e1a181cee2fbd3c2d51e2c7a1e27afa5",
    ),
    (
        "T10TGS.2018190.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T10TGS.2018190.v1.4_merged.tif",
        6295168,
        "2ffd9a5805676b01463aa1df92c1f2e65f7b7c039c90e4a7500a38910ffc6e89",
        "training/subsetted_512x512_HLS.S30.T10TGS.2018190.v1.4.mask.tif",
        525272,
        "1e7480cdb18b3c8d51fd78dd440c434e72d96a713c19276f85632a20054e72f8",
    ),
    (
        "T10UGU.2018245.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T10UGU.2018245.v1.4_merged.tif",
        6295168,
        "fc44dbe9277a06793f8abac67a775d3bb9eb07d0bb4b31e3fb801196e8f62565",
        "training/subsetted_512x512_HLS.S30.T10UGU.2018245.v1.4.mask.tif",
        525272,
        "fd0d2bce787d50350f10e4fb194885d4f70f6c4397595b5db2a86f7bf53af2eb",
    ),
    (
        "T11SMT.2019294.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T11SMT.2019294.v1.4_merged.tif",
        6295168,
        "bf653eb96ee9bd8543ea06e5477be7fd2131d6403872bfcba78ea66423df6805",
        "training/subsetted_512x512_HLS.S30.T11SMT.2019294.v1.4.mask.tif",
        525272,
        "e479eb9e3d16dcd7ecfffeb3c017347222009cb74c95cb44e8785f0606363130",
    ),
    (
        "T11TPE.2019269.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T11TPE.2019269.v1.4_merged.tif",
        6295168,
        "3813f554b8b2d82bc7a26c3fd5e382cd904e4a72f8e7051cd2465bd04c43377d",
        "training/subsetted_512x512_HLS.S30.T11TPE.2019269.v1.4.mask.tif",
        525272,
        "a2876c506fe075263abce00ee409a12b3b888930e764f1147c32f66e85b476ba",
    ),
    (
        "T12STG.2018186.v1",
        "train",
        "validation/subsetted_512x512_HLS.S30.T12STG.2018186.v1.4_merged.tif",
        6295168,
        "6d9ff3c80912699273588a5c13468c7f9b977277d0da23485dd180b047c59ab8",
        "validation/subsetted_512x512_HLS.S30.T12STG.2018186.v1.4.mask.tif",
        525272,
        "799340ad87a0fcacddeedaeff2256ec898ac249dea5475e6a540e39f2907ee10",
    ),
    (
        "T12SYG.2018225.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T12SYG.2018225.v1.4_merged.tif",
        6295168,
        "7d81640e6ac0a6170a8d02d50cbdd2cc237fc9f29e7dbbd7d3294e4487bf7cb4",
        "training/subsetted_512x512_HLS.S30.T12SYG.2018225.v1.4.mask.tif",
        525272,
        "5efc42d679955ff8c665e95fcf15d8d8440dd8eff648025eb2d6ad9a219780c5",
    ),
    (
        "T12TVK.2020308.v1",
        "train",
        "validation/subsetted_512x512_HLS.S30.T12TVK.2020308.v1.4_merged.tif",
        6295168,
        "d726f700c6161238771681aad6be2a2cd140bf02ade68f93092256640c1ab32b",
        "validation/subsetted_512x512_HLS.S30.T12TVK.2020308.v1.4.mask.tif",
        525272,
        "7d815f182de219c2e88e1e9508b84df81ad928eb685104b2d9d532324d3f5c09",
    ),
    (
        "T12TWT.2020276.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T12TWT.2020276.v1.4_merged.tif",
        6295168,
        "678e4f39fac91190ce9fa1a6a13ad30b0a703e79ca202b68df8b7963792b1400",
        "training/subsetted_512x512_HLS.S30.T12TWT.2020276.v1.4.mask.tif",
        525272,
        "dc8bcc47141915172e18228af79155f9c8587b1862a40557dd24d819d754941a",
    ),
    (
        "T13RGP.2020118.v1",
        "train",
        "validation/subsetted_512x512_HLS.S30.T13RGP.2020118.v1.4_merged.tif",
        6295168,
        "94af652d046118fa24805ab01a821d0ec9152f34ec8d42951530cd5dd41b958b",
        "validation/subsetted_512x512_HLS.S30.T13RGP.2020118.v1.4.mask.tif",
        525272,
        "159b9b667f70673d9c85682a2fe016a5f4aef7cd255fbbe9a31783ac6cbe793a",
    ),
    (
        "T13SDT.2019184.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T13SDT.2019184.v1.4_merged.tif",
        6295168,
        "c0868f80ef475b61f3c87542b9173a4b80edd42dacd268fc9547cf49bc6154cf",
        "training/subsetted_512x512_HLS.S30.T13SDT.2019184.v1.4.mask.tif",
        525272,
        "5061eddc74af76014f80eb267ee0b2147b15fe42e484c658aa1aa53cdcbf117f",
    ),
    (
        "T13SFT.2019171.v1",
        "train",
        "validation/subsetted_512x512_HLS.S30.T13SFT.2019171.v1.4_merged.tif",
        6295168,
        "6ab434ca283f9749f58bb58fedda7e5664edc70629b15411192e337846874b16",
        "validation/subsetted_512x512_HLS.S30.T13SFT.2019171.v1.4.mask.tif",
        525272,
        "e2c41cfcf527154244f26ec2cf1483bb86777f172c401ebe5d3fb447d2247f4f",
    ),
    (
        "T14SLE.2019098.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T14SLE.2019098.v1.4_merged.tif",
        6295168,
        "b921e6b6e8bd2a43ed68b4f817e6dd66bf33f67363ff13d36b25649edddbd5be",
        "training/subsetted_512x512_HLS.S30.T14SLE.2019098.v1.4.mask.tif",
        525272,
        "55f34f14371eceece02f843f42e6484d12a9fe6ca84579dd865f9c7472c82b29",
    ),
    (
        "T15RWQ.2021098.v1",
        "train",
        "validation/subsetted_512x512_HLS.S30.T15RWQ.2021098.v1.4_merged.tif",
        6295168,
        "39a8a5270b1ffe8d5bba8b5cc9d0416225f4fa64e236a2bde66c29404992de67",
        "validation/subsetted_512x512_HLS.S30.T15RWQ.2021098.v1.4.mask.tif",
        525272,
        "73a1cf554c7d5539e93fe124d216001713524bb0a36b8b8c30963425f79def08",
    ),
    (
        "T15STA.2018125.v1",
        "train",
        "validation/subsetted_512x512_HLS.S30.T15STA.2018125.v1.4_merged.tif",
        6295168,
        "390a22dde0b506cbda8e96e812881701fccd6af4d16f413aaba13c7e9cfae701",
        "validation/subsetted_512x512_HLS.S30.T15STA.2018125.v1.4.mask.tif",
        525272,
        "f89131607f798f5c621968291e14b024e527a2583a5aeaaa6d733fe2b85c3205",
    ),
    (
        "T15SWV.2018099.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T15SWV.2018099.v1.4_merged.tif",
        6295168,
        "df2765625cd85db00ca90c8f3a8640a0b29a8308e82f2fca12c680c3a921fba6",
        "training/subsetted_512x512_HLS.S30.T15SWV.2018099.v1.4.mask.tif",
        525272,
        "137d6f79ab78762ea30b26503130cf7b19d265f3138c8c9145e47a4fa753c162",
    ),
    (
        "T16RFU.2019250.v1",
        "train",
        "validation/subsetted_512x512_HLS.S30.T16RFU.2019250.v1.4_merged.tif",
        6295168,
        "1ead623c4136059100bc7f1fc39daef43dfe0257e3ac038deb8e40c0152b49e6",
        "validation/subsetted_512x512_HLS.S30.T16RFU.2019250.v1.4.mask.tif",
        525272,
        "f41c8189cdc143f59fd7f4294428a538ca0cce5c5306a50e0cf13b27a7f6666c",
    ),
    (
        "T16SDD.2020093.v1",
        "train",
        "training/subsetted_512x512_HLS.S30.T16SDD.2020093.v1.4_merged.tif",
        6295168,
        "a023147cd40bf80aa15258fe704c62c51f97954021c287ba80d5ef4a9267b228",
        "training/subsetted_512x512_HLS.S30.T16SDD.2020093.v1.4.mask.tif",
        525272,
        "ff1285ca2f447dc2afdc7353222b3d7f66b90ecd0a669349501d7a7c50f0bb3a",
    ),
    (
        "T17SLT.2019112.v1",
        "train",
        "validation/subsetted_512x512_HLS.S30.T17SLT.2019112.v1.4_merged.tif",
        6295168,
        "ef628bf7fad74cad62a052ed9f208486521fa50455519671e37718e16c1ade58",
        "validation/subsetted_512x512_HLS.S30.T17SLT.2019112.v1.4.mask.tif",
        525272,
        "4ee9f13f70aabdf272b8eeaf16142fa9f350389cb1b201271a29dd6dc7fa0b58",
    ),
    (
        "T10TFM.2018110.v1",
        "validation",
        "validation/subsetted_512x512_HLS.S30.T10TFM.2018110.v1.4_merged.tif",
        6295168,
        "bfdac214cc8f46a7da6c4cd905bf38f6d33e54de9f2346df0dc3d3a289918eb4",
        "validation/subsetted_512x512_HLS.S30.T10TFM.2018110.v1.4.mask.tif",
        525272,
        "6c2b42063936b0c6d23d89b507a86bcfa0b6ca62de1ed9afce694ac513200be8",
    ),
    (
        "T11SPV.2020236.v1",
        "validation",
        "training/subsetted_512x512_HLS.S30.T11SPV.2020236.v1.4_merged.tif",
        6295168,
        "322e55a79860b9dd03d43548c805962760a073a446970b714cf7a7295644af57",
        "training/subsetted_512x512_HLS.S30.T11SPV.2020236.v1.4.mask.tif",
        525272,
        "f2cb3adc3a8cfc9f6a9ea058c2219073b897353daac32476ac5d654756b7eea7",
    ),
    (
        "T11TMF.2018222.v1",
        "validation",
        "training/subsetted_512x512_HLS.S30.T11TMF.2018222.v1.4_merged.tif",
        6295168,
        "6b8c89730c1f154b83dbcee1d6a28612a6b0885f664b35e7911daa196107c62e",
        "training/subsetted_512x512_HLS.S30.T11TMF.2018222.v1.4.mask.tif",
        525272,
        "585a5a248ed23cf6d59e81bb8405d4bd7c49a8520902beecb3b713f5d54d2020",
    ),
    (
        "T12RXV.2018217.v1",
        "validation",
        "training/subsetted_512x512_HLS.S30.T12RXV.2018217.v1.4_merged.tif",
        6295168,
        "39f7ca665d41a670a3d1544d281e563650db8829b1e995fa9319fa8e25a1edde",
        "training/subsetted_512x512_HLS.S30.T12RXV.2018217.v1.4.mask.tif",
        525272,
        "11979d66623bc1283ffc0c8cb0d7a7ad16cc467a26c4739095630120121fcc23",
    ),
    (
        "T13TBE.2018220.v1",
        "validation",
        "training/subsetted_512x512_HLS.S30.T13TBE.2018220.v1.4_merged.tif",
        6295168,
        "e85080484fa5882c0307af09faacb6c340a3f66c9c9368003ea377db5baba585",
        "training/subsetted_512x512_HLS.S30.T13TBE.2018220.v1.4.mask.tif",
        525272,
        "3aa03edb6eb327f268c8babd1cd810fed353e63925aed9b885f128a57d9f2c05",
    ),
    (
        "T13TDE.2020247.v1",
        "validation",
        "training/subsetted_512x512_HLS.S30.T13TDE.2020247.v1.4_merged.tif",
        6295168,
        "63eb63e45bb9fe41bd53a41ba078f2a9b056d34cc0fc943b5cab035d03710497",
        "training/subsetted_512x512_HLS.S30.T13TDE.2020247.v1.4.mask.tif",
        525272,
        "33ec6be65d66e2ed106b64229019f1cd307dc8fe0a4690c8047d8b39f88b1a07",
    ),
    (
        "T14SMC.2019258.v1",
        "validation",
        "training/subsetted_512x512_HLS.S30.T14SMC.2019258.v1.4_merged.tif",
        6295168,
        "c7d0b8f4f43602c89a48d0acaa1051f94cb885e1bc965268e85bd5e818bf9d17",
        "training/subsetted_512x512_HLS.S30.T14SMC.2019258.v1.4.mask.tif",
        525272,
        "fc665421e55f94af4c198a2ecd6efc5e7eb5e8df9351e286db5045739da88e40",
    ),
    (
        "T15SWA.2020109.v1",
        "validation",
        "training/subsetted_512x512_HLS.S30.T15SWA.2020109.v1.4_merged.tif",
        6295168,
        "33520267b7f304e06e12777950bb9eba25118cf905bf8782be95628c6ca4a99e",
        "training/subsetted_512x512_HLS.S30.T15SWA.2020109.v1.4.mask.tif",
        525272,
        "41bf346549fa33260c9dddb446606df5b8cc3cb7ea260eada9914e49817b6a42",
    ),
    (
        "T10SDH.2020248.v1",
        "test",
        "training/subsetted_512x512_HLS.S30.T10SDH.2020248.v1.4_merged.tif",
        6295168,
        "256825373fd01eec0da0d341d7b3f683e03d0d71e2f3e58a69f7fc3ba6ad4348",
        "training/subsetted_512x512_HLS.S30.T10SDH.2020248.v1.4.mask.tif",
        525272,
        "ad8ae249875ff8db4ee9e5cfa75b1b38538eba7bfbd4489cb1f9066fd0ef2d27",
    ),
    (
        "T10SEH.2018190.v1",
        "test",
        "validation/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif",
        6295168,
        "13bc592a5e569d837bd8bb3524bb0d2f28418830bcc7b0750e74033078f8b17e",
        "validation/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4.mask.tif",
        525272,
        "67e92de848f4d90730bff525d3aa51fa7c02759e494f264c0d33dcc8c4f6e66a",
    ),
    (
        "T10TFQ.2018245.v1",
        "test",
        "training/subsetted_512x512_HLS.S30.T10TFQ.2018245.v1.4_merged.tif",
        6295168,
        "c5d48a242953f23eeb06a8527204fdbd692150f4fa07cb05952251c64aa78b97",
        "training/subsetted_512x512_HLS.S30.T10TFQ.2018245.v1.4.mask.tif",
        525272,
        "cdfe51739d6bdf6ac2ac57ae307f6182b40256bfc9b8ca8a755c7a6f56c5ebdc",
    ),
    (
        "T10TFT.2018213.v1",
        "test",
        "training/subsetted_512x512_HLS.S30.T10TFT.2018213.v1.4_merged.tif",
        6295168,
        "37190ec2c2cc80eff7a7ad381a20b9a7c14518afffded118393bc56011a96368",
        "training/subsetted_512x512_HLS.S30.T10TFT.2018213.v1.4.mask.tif",
        525272,
        "ded8fbc4a7509e56dc191c54839ba4918a0b50b8e51799de37ff395f6da699f4",
    ),
    (
        "T10TGS.2018245.v1",
        "test",
        "validation/subsetted_512x512_HLS.S30.T10TGS.2018245.v1.4_merged.tif",
        6295168,
        "474fafd750cf9faeefa27f0c9c2d55dfb93c4199c87e8e249d0a7b64b8702377",
        "validation/subsetted_512x512_HLS.S30.T10TGS.2018245.v1.4.mask.tif",
        525272,
        "03a3ba34ab6c205c7fbc45cd5711194a29ffcf0aae3ec737ea9f19490601dd6c",
    ),
    (
        "T10TGT.2018188.v1",
        "test",
        "validation/subsetted_512x512_HLS.S30.T10TGT.2018188.v1.4_merged.tif",
        6295168,
        "dfc3039d03b1ae24260be63af100db7fec80d4185b82ea76914dca9234c0c5c7",
        "validation/subsetted_512x512_HLS.S30.T10TGT.2018188.v1.4.mask.tif",
        525272,
        "4cefbeaf1327e7c75ff3b040c21bc6df5a7290fc1bfa3ec1ed5b5c7196fd3e7e",
    ),
    (
        "T11TPH.2020174.v1",
        "test",
        "training/subsetted_512x512_HLS.S30.T11TPH.2020174.v1.4_merged.tif",
        6295168,
        "c0a7588d7e81c5c1d1ccafb2e6e84f1678e6d0fd927bff7a7246c4e47abe040e",
        "training/subsetted_512x512_HLS.S30.T11TPH.2020174.v1.4.mask.tif",
        525272,
        "9a5cbac806aa96f2b01a97ce3a67dfc98dc5982adf3b2489c4649c3a7549b5ed",
    ),
    (
        "T12SVD.2019183.v1",
        "test",
        "training/subsetted_512x512_HLS.S30.T12SVD.2019183.v1.4_merged.tif",
        6295168,
        "c2a899ad91ef7b81d05727f7baccedbeae93518832ae0fcab0c26d396a553cbe",
        "training/subsetted_512x512_HLS.S30.T12SVD.2019183.v1.4.mask.tif",
        525272,
        "5f2917153a3040c05ecfeda9a171bb4725d7a423471fee52c55639796aa25cd8",
    ),
    (
        "T13REQ.2018156.v1",
        "test",
        "validation/subsetted_512x512_HLS.S30.T13REQ.2018156.v1.4_merged.tif",
        6295168,
        "45583c891a46248d756d50adaf60e2feb246ecedbfcb5f69bf074c1ddf468a62",
        "validation/subsetted_512x512_HLS.S30.T13REQ.2018156.v1.4.mask.tif",
        525272,
        "9f85f2f7e50f40076caa80b0e488268688a3f397b67a671627e6a9fada97f88f",
    ),
    (
        "T13SDV.2020269.v1",
        "test",
        "training/subsetted_512x512_HLS.S30.T13SDV.2020269.v1.4_merged.tif",
        6295168,
        "98bd5b6fb0d4fd436bf0194b28e91edcb9ce0bd24a8ad20ac087a78076ec3832",
        "training/subsetted_512x512_HLS.S30.T13SDV.2020269.v1.4.mask.tif",
        525272,
        "cc10468c737d9126cc7b869e17b23ed04db2fd4705928dfc9ee34d87cb81635e",
    ),
    (
        "T13TDL.2020280.v1",
        "test",
        "validation/subsetted_512x512_HLS.S30.T13TDL.2020280.v1.4_merged.tif",
        6295168,
        "8dcfbf37dbef187f62bd89a72f9e2db5884008b9a1295cbb893a9855b83c7136",
        "validation/subsetted_512x512_HLS.S30.T13TDL.2020280.v1.4.mask.tif",
        525272,
        "625e03be07c054b1629d9183a78f25e36208e86885ace28173de83852b88badd",
    ),
    (
        "T15SXB.2020089.v1",
        "test",
        "validation/subsetted_512x512_HLS.S30.T15SXB.2020089.v1.4_merged.tif",
        6295168,
        "c0bede195f7660c6f484dab92418998920a3644c0567c3cf7e63bd2d9d5278bd",
        "validation/subsetted_512x512_HLS.S30.T15SXB.2020089.v1.4.mask.tif",
        525272,
        "e1f12ef7826981ef0fd2d03901d8bf8dab1e6f537fc9edf75413a4d03215977e",
    ),
)
SAMPLE_LABEL_SOURCE = f"{CORPUS_NAME}; {CORPUS_RELEASE}; {CORPUS_LICENSE}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned_members() -> dict[str, tuple[int, str]]:
    out = {}
    for _name, _role, image, image_bytes, image_sha, label, label_bytes, label_sha in SAMPLE_RECORDS:
        out[image] = (image_bytes, image_sha)
        out[label] = (label_bytes, label_sha)
    return out


def _hub_download_tarball(destination: Path) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(DATASET_ID, TAR_NAME, repo_type="dataset", revision=DATASET_REVISION, local_dir=str(destination.parent))


def fetch_tarball(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> Path:
    """The pinned dataset tarball in the cache, fetched from the Hub at the immutable revision when absent, and
    refused on a size or SHA-256 mismatch (the 2.6 GB file is hashed once per call)."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / TAR_NAME
    if not local.is_file() or local.stat().st_size != TAR_BYTES:
        if fetcher is not None:
            local.write_bytes(fetcher(CORPUS_BASE_URL + TAR_NAME))
        else:
            _hub_download_tarball(local)
    size = local.stat().st_size
    digest = _sha256_file(local)
    if size != TAR_BYTES or digest != TAR_SHA256:
        raise ValueError(f"{TAR_NAME}: {size} bytes with sha256 {digest[:16]}…, pinned {TAR_BYTES} / {TAR_SHA256[:16]}…")
    return local


def extract_pinned_members(tar_path: str | Path, *, cache_dir: str | Path | None = None) -> dict[str, bytes]:
    """Stream through the tarball once and copy out exactly the pinned members (no `extractall`, no paths from
    the archive: each is written under its base name in `cache_dir/chips/`), refusing a size or digest mismatch."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    chips = cache / "chips"
    chips.mkdir(parents=True, exist_ok=True)
    wanted = _pinned_members()
    out: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive:
            if member.name not in wanted or not member.isfile():
                continue
            size, sha = wanted[member.name]
            handle = archive.extractfile(member)
            data = handle.read() if handle is not None else b""
            if len(data) != size or _sha256_bytes(data) != sha:
                raise ValueError(
                    f"{member.name}: {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…, pinned {size} / {sha[:16]}…"
                )
            (chips / Path(member.name).name).write_bytes(data)
            out[member.name] = data
    missing = sorted(set(wanted) - set(out))
    if missing:
        raise ValueError(f"tarball does not contain {len(missing)} pinned members, e.g. {missing[:3]}")
    return out


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, dict[str, bytes]]:
    """Every pinned scene's image and mask bytes, keyed by scene key: from the extracted cache when every file is
    present with its pinned digest, otherwise from the (verified) tarball."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    chips = cache / "chips"
    wanted = _pinned_members()
    cached: dict[str, bytes] = {}
    for member, (size, sha) in wanted.items():
        local = chips / Path(member).name
        if local.is_file() and local.stat().st_size == size:
            data = local.read_bytes()
            if _sha256_bytes(data) == sha:
                cached[member] = data
    if len(cached) != len(wanted):
        cached = extract_pinned_members(fetch_tarball(cache_dir=cache, fetcher=fetcher), cache_dir=cache)
    out = {}
    for name, _role, image, *_rest in SAMPLE_RECORDS:
        label = _rest[2]
        out[name] = {"image": cached[image], "label": cached[label]}
    return out


def read_corpus(files: Mapping[str, Mapping[str, bytes]]) -> dict[str, list[dict[str, Any]]]:
    """Decode the verified bytes into `{id, image, label}` records grouped by role (train / validation / test)."""
    import tempfile

    splits: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    for name, role, image_member, *_rest in SAMPLE_RECORDS:
        if name not in files:
            raise ValueError(f"corpus is missing {name}")
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.tif"
            label_path = Path(tmp) / "label.tif"
            image_path.write_bytes(files[name]["image"])
            label_path.write_bytes(files[name]["label"])
            image = read_chip(image_path)
            label = read_mask(label_path)
        raw = {
            "id": f"{role}-{len(splits[role]):03d}",
            "source_id": name,
            "region": name.split(".")[0],  # the HLS tile id (UTM zone + grid square)
            "split": role,
            "image": image,
            "label": label,
            "source": f"{CORPUS_BASE_URL}{TAR_NAME}#{image_member}",
        }
        splits[role].append(check_record(raw))  # no-data replaced, range checked, label checked
    return splits


def fetch_sample_dataset(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus (roles from the model repository's splits)."""
    return read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no chip (by pixel digest) appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = chip_digest(record)
            if key in seen and seen[key] != name:
                raise ValueError(f"chip {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train / validation / test after de-duplicating chips. Chips from one
    fire or one tile are near-duplicates; group them yourself (one fire per split) when that matters."""
    import random

    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = chip_digest(record)
        if key not in seen:
            seen.add(key)
            unique.append(record)
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_test = max(1, round(len(unique) * test_fraction))
    n_val = round(len(unique) * val_fraction)
    splits = {"test": unique[:n_test], "validation": unique[n_test : n_test + n_val], "train": unique[n_test + n_val :]}
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(f"split leaves {len(splits['train'])} training chips; at least {MIN_RECORDS} are required")
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, image, label}` records from a directory or a zip holding `pairs.csv` (columns `id`, `image`,
    `label`) beside six-band 512 × 512 GeoTIFF chips and single-band label rasters; files are decoded from bytes,
    never extracted to disk."""
    import tempfile

    source = Path(path)
    if source.is_dir():
        table = (source / "pairs.csv").read_text(encoding="utf-8")
        loader = lambda name: (source / name).read_bytes()  # noqa: E731
    elif source.is_file() and source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        members = {Path(n).name: n for n in archive.namelist()}
        if "pairs.csv" not in members:
            raise ValueError("BYOD zip must contain pairs.csv")
        table = archive.read(members["pairs.csv"]).decode("utf-8")
        loader = lambda name: archive.read(members[name])  # noqa: E731
    else:
        raise ValueError("BYOD datasets must be a directory or a .zip holding pairs.csv and the GeoTIFF files")
    rows = list(csv.DictReader(io.StringIO(table)))
    missing = {"id", "image", "label"} - set(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"pairs.csv is missing columns {sorted(missing)}")
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for row in rows:
            image_path = Path(tmp) / "image.tif"
            image_path.write_bytes(loader(row["image"]))
            record: dict[str, Any] = {"id": row["id"], "image": read_chip(image_path)}
            if row.get("label"):
                label_path = Path(tmp) / "label.tif"
                label_path.write_bytes(loader(row["label"]))
                record["label"] = read_mask(label_path)
            out.append(record)
    return out


def write_sample_pair(record: Mapping[str, Any], image_path: str | Path, label_path: str | Path) -> dict[str, str]:
    """Write one record as a six-band float32 TIFF and a single-band int16 TIFF (the BYOD shape, without
    georeferencing) and return both paths."""
    import numpy as np
    import tifffile

    image_out, label_out = Path(image_path), Path(label_path)
    image_out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(image_out, np.asarray(record["image"], dtype=np.float32), photometric="minisblack", planarconfig="separate")
    tifffile.imwrite(label_out, np.asarray(record["label"], dtype=np.int16), photometric="minisblack")
    return {"image": str(image_out), "label": str(label_out)}


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the pairs table of a split (id, image, label, provenance) in the shape BYOD expects."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image", "label", "region", "source"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record["id"],
                    "image": f"{record.get('source_id', record['id'])}_merged.tif",
                    "label": f"{record.get('source_id', record['id'])}.mask.tif",
                    "region": record.get("region", ""),
                    "source": record.get("source", ""),
                }
            )
    return out


def dataset_manifest(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Validate every split and summarise the dataset (counts, class balance, digests) for provenance exports."""
    summary: dict[str, Any] = {"model_id": MODEL_ID, "image_size": IMAGE_SIZE, "splits": {}}
    for name, records in splits.items():
        report = validate_dataset(records, min_records=1)
        summary["splits"][name] = {
            "n_records": report["n_records"],
            "class_pixel_fraction": report["class_pixel_fraction"],
            "ignored_pixels": report["ignored_pixels"],
            "regions": sorted({str(r.get("region", "")) for r in records if r.get("region")}),
            "digest": report["digest"],
        }
    summary["disjoint"] = check_split_disjoint(splits)
    digests = json.dumps({k: v["digest"] for k, v in summary["splits"].items()}, sort_keys=True)
    summary["digest"] = hashlib.sha256(digests.encode("utf-8")).hexdigest()
    return summary
