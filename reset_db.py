import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from getpass import getpass

print("⚠️  This will DELETE the 'slaq_d_db' database and all its data.")
password = getpass("Enter your PostgreSQL password to continue: ")

try:
    # Connect to the default 'postgres' database to perform administrative tasks
    conn = psycopg2.connect(
        dbname='postgres', 
        user='postgres', 
        password=password, 
        host='localhost', 
        port='5432'
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    # Force drop the database (kills active connections like the hung runserver)
    print("Dropping database...")
    cur.execute("DROP DATABASE IF EXISTS slaq_d_db WITH (FORCE);")
    
    print("✅ Database 'slaq_d_db' successfully deleted.")
    cur.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")