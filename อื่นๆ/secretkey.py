import os
import secrets
from dotenv import load_dotenv

def generate_secret_key():
    """Generate a secure random secret key."""
    return secrets.token_hex(32)  # 64-character hex string

def create_env_file():
    """
    Create a .env file with secure, randomly generated keys.
    
    NOTE: This should ONLY be used during initial setup.
    Never commit this file to version control.
    """
    # Generate secure keys
    secret_key = generate_secret_key()
    csrf_secret_key = generate_secret_key()
    
    # Prepare environment configuration
    env_config = [
        f"SECRET_KEY={secret_key}",
        f"WTF_CSRF_SECRET_KEY={csrf_secret_key}",
        "FLASK_ENV=development",
        "SESSION_LIFETIME=30",
        "UPLOAD_FOLDER=uploads",
        "DATABASE_URL=sqlite:///your_database.db",  # Example database URL
        # Add other sensitive configurations here
    ]
    
    # Write to .env file
    with open('.env', 'w') as env_file:
        env_file.write('\n'.join(env_config))
    
    print("🔐 .env file generated successfully!")
    print("IMPORTANT: Keep this file secret and do not commit to version control.")

def load_environment_config():
    """
    Load environment configurations securely.
    Use python-dotenv to load .env file in development.
    """
    # Check if .env file exists
    if not os.path.exists('.env'):
        # If .env doesn't exist, create it
        create_env_file()
    
    # Load .env file 
    load_dotenv()
    
    # Additional security checks
    secret_key = os.getenv('SECRET_KEY')
    if not secret_key or len(secret_key) < 64:
        # If secret key is invalid, create a new .env file
        print("⚠️ Invalid SECRET_KEY. Regenerating...")
        create_env_file()
        load_dotenv()  # Reload environment variables

def get_database_url():
    """
    Securely retrieve database URL with fallback and validation.
    """
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        # Fallback for development, but warn about security
        print("⚠️ WARNING: Using default database URL. Set DATABASE_URL in production!")
        return 'sqlite:///dev_database.db'
    
    return database_url

# Usage example in your main app setup
if __name__ == '__main__':
    # Load environment configurations
    load_environment_config()
    
    # Get database URL
    database_url = get_database_url()
    
    # Print out some configurations (for demonstration)
    print(f"Database URL: {database_url}")
    print(f"Secret Key Length: {len(os.getenv('SECRET_KEY', ''))}")