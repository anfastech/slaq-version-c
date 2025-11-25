import os
import sys
import shutil
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / '.env'

def read_env():
    config = {}
    if ENV_FILE.exists():
        with open(ENV_FILE, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    config[key] = value
    return config

def fix_env_file():
    print("🔧 Checking .env configuration...")
    if not ENV_FILE.exists():
        print("❌ .env file missing!")
        return None

    # Read current lines
    with open(ENV_FILE, 'r') as f:
        lines = f.readlines()

    new_lines = []
    db_fixed = False
    
    for line in lines:
        # Fix Database Name
        if line.startswith('DB_NAME='):
            new_lines.append('DB_NAME=slaq_d_db\n')
            db_fixed = True
        else:
            new_lines.append(line)
            
    if not db_fixed:
        new_lines.append('DB_NAME=slaq_d_db\n')

    # Write back
    with open(ENV_FILE, 'w') as f:
        f.writelines(new_lines)
    
    print("✅ .env configuration normalized (DB_NAME=slaq_d_db)")
    return read_env()

def reset_database(config):
    db_name = config.get('DB_NAME', 'slaq_d_db')
    db_user = config.get('DB_USER', 'postgres')
    db_pass = config.get('DB_USER_PASSWORD', '')
    
    print(f"\n🗑️  Resetting Database: {db_name}...")
    
    try:
        # Connect to 'postgres' system db
        conn = psycopg2.connect(
            dbname='postgres', 
            user=db_user, 
            password=db_pass, 
            host='localhost', 
            port='5432'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Drop both potential databases to be safe
        cur.execute(f"DROP DATABASE IF EXISTS {db_name} WITH (FORCE);")
        cur.execute(f"DROP DATABASE IF EXISTS slaq_db WITH (FORCE);") # Clean up the wrong one too
        
        # Create fresh
        cur.execute(f"CREATE DATABASE {db_name};")
        
        print(f"✅ Database '{db_name}' created fresh.")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Database Error: {e}")
        print("   Check your DB_USER_PASSWORD in .env")
        return False

def clean_migrations():
    print("\n🧹 Cleaning migration files...")
    apps = ['core', 'diagnosis']
    for app in apps:
        mig_dir = PROJECT_ROOT / app / 'migrations'
        if mig_dir.exists():
            for file in mig_dir.glob('0*.py'):
                if file.name != '__init__.py':
                    print(f"   - Deleting {file.name}")
                    os.remove(file)
    print("✅ Old migrations removed.")

def run_django_setup():
    print("\n🏗️  Building Django Tables...")
    
    # Make Migrations
    if os.system(f"{sys.executable} manage.py makemigrations core diagnosis") != 0:
        print("❌ Make Migrations failed")
        return

    # Migrate
    if os.system(f"{sys.executable} manage.py migrate") != 0:
        print("❌ Migrate failed")
        return
        
    print("✅ Database schema applied.")
    
    print("\n👤 Create Superuser (Admin)")
    os.system(f"{sys.executable} manage.py createsuperuser")

if __name__ == "__main__":
    print("="*50)
    print("SLAQ SYSTEM REBUILDER")
    print("="*50)
    
    # 1. Fix Environment
    config = fix_env_file()
    if not config: sys.exit(1)
    
    # 2. Reset DB
    if reset_database(config):
        # 3. Clean Files
        clean_migrations()
        
        # 4. Build Django
        run_django_setup()
        
        print("\n" + "="*50)
        print("✅ SYSTEM REBUILT SUCCESSFULLY")
        print("="*50)
        print("1. Restart Celery: celery -A slaq_project worker --pool=solo -l info")
        print("2. Restart Server: python manage.py runserver")
    else:
        print("\n❌ Aborted due to database error.")