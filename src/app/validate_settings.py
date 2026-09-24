"""
validate all config setting

"""
from src.app.config import get_settings


def main():
    settings = get_settings()
    print(f"Enviroment: {settings.app_env}")
    print(f"primary model: {settings.primary_model}")
    print(f'Rate Limit: {settings.rate_limit}')
    print(f'Cahe TTL: {settings.cache_ttl_seconds}s')
    print(f'Max retries: {settings.max_retries}')
    print(f'is production: {settings.is_production}')
    print("Config loaded successfullly!")
    
    
if __name__ == "__main__":
    main()