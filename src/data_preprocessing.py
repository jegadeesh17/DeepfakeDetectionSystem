import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data_ingestion import ingest_data, resolve_image_path

LABEL_MAPPING = {"REAL": 0, "FAKE": 1}


class DeepfakeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = resolve_image_path(row["local_path"])
        label_str = str(row["label"]).upper()

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = LABEL_MAPPING.get(label_str)
        if label is None:
            raise ValueError(f"Unknown label '{label_str}' at index {idx}.")

        return image, torch.tensor(label, dtype=torch.float32)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def get_inference_transform():
    _, val_transform = get_transforms()
    return val_transform


def clean_data(df):
    if df is not None and not df.empty:
        df = df.dropna(subset=["local_path", "label"]).copy()
    return df


def get_dataloaders(
    batch_size=8,
    val_frac=0.1,
    test_frac=0.2,
    random_state=42,
    force_rebuild=False,
):
    """Return train/val/test loaders with a 70/10/20 stratified split by default."""
    torch.manual_seed(random_state)

    df = ingest_data(force_rebuild=force_rebuild)
    if df is None or df.empty:
        raise ValueError("Data could not be loaded or ingested.")

    df = clean_data(df)

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_frac,
        random_state=random_state,
        stratify=df["label"],
    )
    val_size = val_frac / (1 - test_frac)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_df["label"],
    )

    train_transform, val_transform = get_transforms()
    train_dataset = DeepfakeDataset(train_df, transform=train_transform)
    val_dataset = DeepfakeDataset(val_df, transform=val_transform)
    test_dataset = DeepfakeDataset(test_df, transform=val_transform)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
