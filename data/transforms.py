"""
Mammography Image Transforms
=============================
Eğitim ve doğrulama/test için görüntü dönüşüm pipeline'ları.

16-bit PNG desteği:
- Görüntüler dataset.py'de numpy→tensor olarak [0,1] aralığına normalize edilir.
- Bu modül SADECE spatial augmentation ve channel normalization yapar.
- Dataset varyantına (seg/noseg) göre farklı normalizasyon istatistikleri kullanılır.

Mamografi görüntüleri için özel dikkat edilmesi gerekenler:
- Dikey çevirme genelde yapılmaz (anatomik yönelimi bozar).
- Aşırı renk/kontrast değişikliği diagnostik bilgiyi bozabilir.
- Grayscale görüntüler 3 kanala kopyalandığı için, 3 kanala da aynı
  normalizasyon değeri uygulanmalıdır.
"""

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF


# 16-bit dataset normalizasyon istatistikleri (all-pixel, 0-1 scale)
# Eski Dataset_512 / Dataset_512_Test dataseti için
DATASET_STATS = {
    "seg": {
        "mean": [0.0921, 0.0921, 0.0921],
        "std":  [0.1520, 0.1520, 0.1520],
    },
    "noseg": {
        "mean": [0.0718, 0.0718, 0.0718],
        "std":  [0.1499, 0.1499, 0.1499],
    },
}

# 8-bit dataset normalizasyon istatistikleri (all-pixel, 0-1 scale)
# Dataset_1024_8bit dataseti için (claude.md Section 3, compute_norm_stats.py)
# CLAHE + letterbox 1024x1024 + 8-bit PNG pipeline.
# NOT: Eski BIRADS-Full-Train-8Bit-Processed değerleri (0.0990/0.1644) YANLIŞ idi.
DATASET_STATS_8BIT = {
    "noseg": {
        "mean": [0.1210, 0.1210, 0.1210],
        "std":  [0.1977, 0.1977, 0.1977],
    },
}

# Fallback: hiçbir eşleşme yoksa (genişletilebilirlik için korunur)
IMAGENET_MEAN = [0.449, 0.449, 0.449]
IMAGENET_STD = [0.226, 0.226, 0.226]


def _get_norm_stats(data_cfg: dict) -> tuple:
    """
    Config'den uygun normalizasyon istatistiklerini döndürür.

    16-bit modda Dataset_512 istatistikleri (DATASET_STATS) kullanılır.
    8-bit modda BIRADS-Full-Train-8Bit-Processed istatistikleri (DATASET_STATS_8BIT) kullanılır.
    """
    bit_depth = data_cfg.get("bit_depth", 8)
    variant = data_cfg.get("dataset_variant", "seg")

    if bit_depth == 16 and variant in DATASET_STATS:
        stats = DATASET_STATS[variant]
        return stats["mean"], stats["std"]
    elif bit_depth == 8 and variant in DATASET_STATS_8BIT:
        stats = DATASET_STATS_8BIT[variant]
        return stats["mean"], stats["std"]
    else:
        return IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(data_cfg: dict) -> torch.nn.Module:
    """
    Eğitim seti için augmentation pipeline'ı oluşturur.

    16-bit modda girdi zaten (3, H, W) float32 tensor olarak gelir.
    Transform'lar torchvision.transforms.v2 ile tensor üzerinde çalışır.
    """
    aug = data_cfg.get("augmentation", {})
    img_size = data_cfg["image_size"]
    mean, std = _get_norm_stats(data_cfg)

    transform_list = [
        T.Resize((img_size, img_size), antialias=True),
    ]

    if aug.get("enabled", True):
        if aug.get("horizontal_flip", 0) > 0:
            transform_list.append(
                T.RandomHorizontalFlip(p=aug["horizontal_flip"])
            )

        if aug.get("rotation_degrees", 0) > 0:
            transform_list.append(
                T.RandomRotation(degrees=aug["rotation_degrees"])
            )

        brightness = aug.get("brightness", 0)
        contrast = aug.get("contrast", 0)
        if brightness > 0 or contrast > 0:
            transform_list.append(
                T.ColorJitter(brightness=brightness, contrast=contrast)
            )

        transform_list.append(
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            )
        )

    # Normalizasyon (tensor üzerinde çalışır)
    transform_list.append(T.Normalize(mean=mean, std=std))

    # Random erasing
    if aug.get("enabled", True) and aug.get("random_erasing", 0) > 0:
        transform_list.append(
            T.RandomErasing(p=aug["random_erasing"])
        )

    return T.Compose(transform_list)


def get_val_transforms(data_cfg: dict) -> torch.nn.Module:
    """
    Doğrulama ve test seti için transform pipeline'ı.
    Augmentation uygulanmaz, sadece resize ve normalize yapılır.
    """
    img_size = data_cfg["image_size"]
    mean, std = _get_norm_stats(data_cfg)

    return T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.Normalize(mean=mean, std=std),
    ])


def get_inverse_normalize(data_cfg: dict = None) -> T.Normalize:
    """
    Normalize işlemini tersine çevirir (Grad-CAM görselleştirmesi için).
    """
    if data_cfg is not None:
        mean, std = _get_norm_stats(data_cfg)
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1.0 / s for s in std]
    return T.Normalize(mean=inv_mean, std=inv_std)
