import os
import json
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DataDragon:
    """
    Interface to Riot's DataDragon API for static game data.
    """
    BASE_URL = "https://ddragon.leagueoflegends.com"
    
    def __init__(self, cache_dir="data/ddragon"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.version = self._get_latest_version()
        self.champion_data: Dict[str, Any] = {}
        self.item_data: Dict[str, Any] = {}
        
        # Load or fetch data
        self._load_data()
        
    def _get_latest_version(self) -> str:
        """Fetch latest patch version."""
        try:
            versions = requests.get(f"{self.BASE_URL}/api/versions.json").json()
            return versions[0]
        except Exception as e:
            logger.warning(f"Failed to fetch DDragon version: {e}. Using fallback '14.1.1'.")
            return "14.1.1"
            
    def _load_data(self):
        """Load data from cache or fetch from API."""
        champ_path = os.path.join(self.cache_dir, f"champion_{self.version}.json")
        item_path = os.path.join(self.cache_dir, f"item_{self.version}.json")
        
        if os.path.exists(champ_path):
            with open(champ_path, 'r') as f:
                self.champion_data = json.load(f)
        else:
            logger.info(f"Fetching champion data for {self.version}...")
            url = f"{self.BASE_URL}/cdn/{self.version}/data/en_US/championFull.json"
            try:
                data = requests.get(url).json()
                self.champion_data = data['data']
                with open(champ_path, 'w') as f:
                    json.dump(self.champion_data, f)
            except Exception as e:
                logger.error(f"Failed to fetch champions: {e}")
                
        if os.path.exists(item_path):
            with open(item_path, 'r') as f:
                self.item_data = json.load(f)
        else:
            logger.info(f"Fetching item data for {self.version}...")
            url = f"{self.BASE_URL}/cdn/{self.version}/data/en_US/item.json"
            try:
                data = requests.get(url).json()
                self.item_data = data['data']
                with open(item_path, 'w') as f:
                    json.dump(self.item_data, f)
            except Exception as e:
                logger.error(f"Failed to fetch items: {e}")

    def get_champion_context(self, champion_name: str) -> str:
        """
        Get concise context for a champion (Role, Passive, R).
        Returns string suitable for LLM prompt context.
        """
        # DataDragon keys are usually PascalCase, but inputs might vary.
        # We try strict match, then spacing removal.
        # "Lee Sin" -> key "LeeSin"
        key = champion_name.replace(" ", "").replace("'", "").capitalize() # Approximate key normalize
        
        # Special cases mapping (Wukong -> MonkeyKing)
        mapping = {"Wukong": "MonkeyKing", "RenataGlasc": "Renata", "KogMaw": "KogMaw", "RekSai": "RekSai"}
        if champion_name in mapping: key = mapping[champion_name]
        
        # Try finding case-insensitive
        real_key = None
        for k in self.champion_data.keys():
            if k.lower() == key.lower() or k.lower() == champion_name.replace(" ", "").lower():
                real_key = k
                break
        
        if not real_key:
            return f"{champion_name}: (Assessment unavailable - Not found in DB)"
            
        c = self.champion_data[real_key]
        name = c['name']
        title = c['title']
        tags = "/".join(c['tags'])
        passive = c['passive']['name']
        
        # Spells: Q, W, E, R
        spells = c['spells']
        q = spells[0]['name']
        w = spells[1]['name']
        e = spells[2]['name']
        r = spells[3]['name']
        r_desc = spells[3]['description'] # Ultimate description is most critical
        
        # Clean description (remove tags like <br>)
        import re
        r_desc_clean = re.sub(r'<[^>]+>', '', r_desc)
        
        return (f"{name} ({tags}) - {title}\n"
                f"   Passive: {passive}\n"
                f"   R (Ultimate): {r}: {r_desc_clean[:100]}...") # Truncate for token efficiency

class GameKnowledge:
    """Wrapper for semantic access to static data."""
    _instance = None
    
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = DataDragon()
        return cls._instance

if __name__ == "__main__":
    # Test fetch
    logging.basicConfig(level=logging.INFO)
    dd = DataDragon()
    print(dd.get_champion_context("Jinx"))
    print(dd.get_champion_context("Lee Sin"))
