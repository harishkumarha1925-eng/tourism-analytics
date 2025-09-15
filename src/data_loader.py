# src/data_loader.py
from pathlib import Path
import pandas as pd
import zipfile
import tempfile
import os

def _load_excel_safe(path: Path):
    if path.exists():
        try:
            return pd.read_excel(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def merge_updated_item(data_dir: str = "data", write_merged: bool = True):
    """
    Merge Item.xlsx and Updated_Item.xlsx. If the data_dir doesn't exist, create it.
    If both source files are empty/missing, returns empty DataFrame and does not save.
    The write_merged flag controls whether to attempt to save Item_merged.xlsx.
    """
    p = Path(data_dir)
    orig = _load_excel_safe(p / "Item.xlsx")
    updated = _load_excel_safe(p / "Updated_Item.xlsx")

    # If both empty, return empty DF
    if (orig is None or orig.empty) and (updated is None or updated.empty):
        return pd.DataFrame()

    if updated is None or updated.empty:
        merged = orig.copy()
    else:
        if "AttractionId" not in updated.columns:
            raise ValueError("Updated_Item.xlsx must contain 'AttractionId' column")
        updated = updated.drop_duplicates(subset=["AttractionId"], keep="last")
        if not orig.empty and "AttractionId" in orig.columns:
            orig_without_updated = orig[~orig["AttractionId"].isin(updated["AttractionId"])].copy()
            merged = pd.concat([orig_without_updated, updated], ignore_index=True, sort=False)
        else:
            merged = updated.copy()

    if write_merged:
        # Ensure directory exists, else attempt to save to a safe temp directory
        try:
            p.mkdir(parents=True, exist_ok=True)
            merged_path = p / "Item_merged.xlsx"
            merged.to_excel(merged_path, index=False)
            # return merged and path optionally
            return merged
        except Exception as e:
            # fallback: try to save in the system temporary directory
            try:
                tmp_path = Path(tempfile.gettempdir()) / "Item_merged.xlsx"
                merged.to_excel(tmp_path, index=False)
                return merged
            except Exception:
                # final fallback: do not save, just return merged DF
                return merged
    else:
        return merged

def unzip_additional_zip(data_dir: str = "data", zip_name: str = None):
    p = Path(data_dir)
    if zip_name is None:
        zips = list(p.glob("Additional_Data_for_Attraction_Sites*.zip"))
        if not zips:
            return []
        zip_path = zips[0]
    else:
        zip_path = p / zip_name
        if not zip_path.exists():
            return []
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(p)
    return [f.name for f in p.iterdir() if f.is_file()]

def load_raw(data_dir: str = "data"):
    p = Path(data_dir)
    # If no data dir on server, don't fail; create it (safer) but don't assume files exist.
    if not p.exists():
        # create directory so code that tries to save merged file won't fail
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            # if cannot create (rare), proceed anyway — merge_updated_item will handle fallback
            pass

    # Make sure merged item is present or attempt to create it (no crash)
    merged_item_path = p / "Item_merged.xlsx"
    if not merged_item_path.exists():
        _ = merge_updated_item(data_dir, write_merged=True)

    tx = _load_excel_safe(p / "Transaction.xlsx")
    users = _load_excel_safe(p / "User.xlsx")
    items = _load_excel_safe(merged_item_path) if merged_item_path.exists() else _load_excel_safe(p / "Item_merged.xlsx")
    # if still empty, fall back to raw files (if any)
    if (items is None or items.empty):
        items = _load_excel_safe(p / "Updated_Item.xlsx")
        if items is None or items.empty:
            items = _load_excel_safe(p / "Item.xlsx")

    city = _load_excel_safe(p / "City.xlsx")
    country = _load_excel_safe(p / "Country.xlsx")
    continent = _load_excel_safe(p / "Continent.xlsx")
    region = _load_excel_safe(p / "Region.xlsx")
    atype = _load_excel_safe(p / "Type.xlsx")
    mode = _load_excel_safe(p / "Mode.xlsx")

    return {
        "tx": tx, "users": users, "items": items, "city": city,
        "country": country, "continent": continent, "region": region,
        "type": atype, "mode": mode
    }

def build_consolidated(dfs: dict):
    """
    Join transaction + user + item tables into consolidated visit-level DataFrame.
    This function expects dfs is the dict returned by load_raw(). It is defensive:
    if required tables are missing, it returns an empty DataFrame.
    """
    tx = dfs.get("tx", pd.DataFrame())
    users = dfs.get("users", pd.DataFrame())
    items = dfs.get("items", pd.DataFrame())
    city = dfs.get("city", pd.DataFrame())
    country = dfs.get("country", pd.DataFrame())
    region = dfs.get("region", pd.DataFrame())
    continent = dfs.get("continent", pd.DataFrame())
    atype = dfs.get("type", pd.DataFrame())
    mode = dfs.get("mode", pd.DataFrame())

    if tx.empty:
        # Nothing to build
        return pd.DataFrame()

    # mapping dicts (defensive)
    city_name = dict(zip(city["CityId"], city["CityName"])) if not city.empty and "CityId" in city.columns and "CityName" in city.columns else {}
    country_name = dict(zip(country["CountryId"], country["Country"])) if not country.empty and "CountryId" in country.columns and "Country" in country.columns else {}
    region_name = dict(zip(region["RegionId"], region["Region"])) if not region.empty and "RegionId" in region.columns and "Region" in region.columns else {}
    continent_name = dict(zip(continent["ContinentId"], continent["Continent"])) if not continent.empty and "ContinentId" in continent.columns and "Continent" in continent.columns else {}
    atype_map = dict(zip(atype["AttractionTypeId"], atype["AttractionType"])) if not atype.empty and "AttractionTypeId" in atype.columns else {}
    mode_map = dict(zip(mode["VisitModeId"], mode["VisitMode"])) if not mode.empty and "VisitModeId" in mode.columns else {}
    city_to_country = dict(zip(city["CityId"], city["CountryId"])) if not city.empty and "CityId" in city.columns and "CountryId" in city.columns else {}

    # users mapping
    if not users.empty:
        if "CityId" in users.columns:
            users["UserCityName"] = users["CityId"].map(city_name)
        if "CountryId" in users.columns:
            users["UserCountry"] = users["CountryId"].map(country_name)
        if "RegionId" in users.columns:
            users["UserRegion"] = users["RegionId"].map(region_name)
        if "ContinentId" in users.columns:
            users["UserContinent"] = users["ContinentId"].map(continent_name)

    # items mapping
    if not items.empty:
        if "AttractionCityId" in items.columns:
            items["AttractionCityName"] = items["AttractionCityId"].map(city_name)
            items["AttractionCountryId"] = items["AttractionCityId"].map(city_to_country)
            items["AttractionCountry"] = items["AttractionCountryId"].map(country_name)
        if "AttractionTypeId" in items.columns:
            items["AttractionType"] = items["AttractionTypeId"].map(atype_map)

    if "VisitMode" in tx.columns:
        tx["VisitModeName"] = tx["VisitMode"].map(mode_map)

    df = tx.merge(users, on="UserId", how="left")
    df = df.merge(items, on="AttractionId", how="left")

    if "Rating" in df.columns:
        df = df.dropna(subset=["UserId", "AttractionId", "Rating"])
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        df = df.dropna(subset=["Rating"])

    return df
