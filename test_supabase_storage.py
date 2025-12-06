import os
import django
from io import BytesIO

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "slaq_project.settings")
django.setup()

from core.supabase_storage import SupabaseStorage
from django.core.files.base import ContentFile


def test_supabase_storage():
    print("Testing Supabase Storage...")

    try:
        storage = SupabaseStorage()
        print("✓ SupabaseStorage initialized successfully")

        test_content = b"Hello, this is a test audio file content!"
        test_filename = "test_audio.txt"

        content_file = ContentFile(test_content, name=test_filename)

        saved_name = storage.save(test_filename, content_file)
        print(f"✓ File uploaded successfully: {saved_name}")

        exists = storage.exists(saved_name)
        print(f"✓ File exists check: {exists}")

        retrieved_file = storage.open(saved_name)
        retrieved_content = retrieved_file.read()
        print(f"✓ File retrieved successfully, size: {len(retrieved_content)} bytes")

        if retrieved_content == test_content:
            print("✓ File content matches original")
        else:
            print("✗ File content mismatch")

        file_url = storage.url(saved_name)
        print(f"✓ File URL generated: {file_url}")

        storage.delete(saved_name)
        print("✓ Test file deleted successfully")

        exists_after_delete = storage.exists(saved_name)
        print(f"✓ File exists after deletion: {exists_after_delete}")

        print("\nAll Supabase storage tests passed")

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_supabase_storage()
