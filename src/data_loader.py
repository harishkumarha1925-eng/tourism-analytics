# src/data_loader.py
from pathlib import Path
import pandas as pd
import zipfile

def _load_excel_safe(path: Path):
    if path.exists():
        return pd.read_excel(path)
    return pd.DataFrame()

def merge_updated_item(data_dir: str = "data"):
    """
    If Updated_Item.xlsx exists, merge it with Item.xlsx.
    Updated rows override original rows with same AttractionId.
    Save result to Item_merged.xlsx and return DataFrame.
    """
    p = Path(data_dir)
    orig = _load_excel_safe(p / "Item.xlsx")
    updated = _load_excel_safe(p / "Updated_Item.xlsx")

    if updated is None or updated.empty:
        # No updated file present, use original
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

    merged_path = p / "Item_merged.xlsx"
    merged.to_excel(merged_path, index=False)
    return merged

def unzip_additional_zip(data_dir: str = "data", zip_name: str = None):
    """
    Optional: Unzip Additional_Data_for_Attraction_Sites...zip into data/
    """
    p = Path(data_dir)
    if zip_name is None:
        # try to find any zip starting with Additional_Data_for_Attraction_Sites
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
    # ensure merged item file exists
    merged_item = p / "Item_merged.xlsx"
    if not merged_item.exists():
        _ = merge_updated_item(data_dir)

    tx = _load_excel_safe(p / "Transaction.xlsx")
    users = _load_excel_safe(p / "User.xlsx")
    items = _load_excel_safe(merged_item)
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
    """
    tx = dfs["tx"].copy()
    users = dfs["users"].copy()
    items = dfs["items"].copy()
    city = dfs["city"].copy()
    country = dfs["country"].copy()
    region = dfs["region"].copy()
    continent = dfs["continent"].copy()
    atype = dfs["type"].copy()
    mode = dfs["mode"].copy()

    # defensive mapping dicts
    city_name = dict(zip(city["CityId"], city["CityName"])) if not city.empty else {}
    country_name = dict(zip(country["CountryId"], country["Country"])) if not country.empty else {}
    region_name = dict(zip(region["RegionId"], region["Region"])) if not region.empty else {}
    continent_name = dict(zip(continent["ContinentId"], continent["Continent"])) if not continent.empty else {}
    atype_map = dict(zip(atype["AttractionTypeId"], atype["AttractionType"])) if not atype.empty else {}
    mode_map = dict(zip(mode["VisitModeId"], mode["VisitMode"])) if not mode.empty else {}
    city_to_country = dict(zip(city["CityId"], city["CountryId"])) if not city.empty else {}

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

    # merge
    df = tx.merge(users, on="UserId", how="left")
    df = df.merge(items, on="AttractionId", how="left")

    # minimal cleaning for Rating
    if "Rating" in df.columns:
        df = df.dropna(subset=["UserId", "AttractionId", "Rating"])
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        df = df.dropna(subset=["Rating"])

    return df
