
from pathlib import Path
import re
import uuid

# Main video directory
RAW_FOLDER = Path("dataset/raw_videos")

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"
}

DRY_RUN = False

CLASSES = ["clear"]


def rename_videos():
    for class_name in CLASSES:
        folder = RAW_FOLDER / class_name

        if not folder.exists():
            print(f"\n[WARNING] Folder not found: {folder}")
            continue

        # Find videos
        videos = [
            file for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in VIDEO_EXTENSIONS
        ]

        # Identify files that already have the correct numbered format
        numbered_pattern = re.compile(
            rf"^{re.escape(class_name)}_(\d+)$",
            re.IGNORECASE
        )

        existing_numbered = []
        new_videos = []

        for video in videos:
            match = numbered_pattern.match(video.stem)

            if match:
                existing_numbered.append(
                    (int(match.group(1)), video)
                )
            else:
                new_videos.append(video)

        # Find the highest existing number
        highest_number = max(
            (number for number, _ in existing_numbered),
            default=0
        )

        # Sort only the new videos
        new_videos.sort(key=lambda x: x.name.lower())

        print(f"\n{'=' * 50}")
        print(f"{class_name.upper()} RENAME SUMMARY")
        print(f"{'=' * 50}")

        print(f"Existing numbered videos: {len(existing_numbered)}")
        print(f"New videos to rename: {len(new_videos)}")
        print(f"Highest existing number: {highest_number}")

        if not new_videos:
            print("No new videos to rename.")
            continue

        # Assign new numbers after the highest existing number
        rename_pairs = []

        for index, old_path in enumerate(
            new_videos,
            start=highest_number + 1
        ):
            new_name = (
                f"{class_name}_{index:03d}"
                f"{old_path.suffix.lower()}"
            )

            new_path = folder / new_name

            if new_path.exists():
                raise FileExistsError(
                    f"Target already exists: {new_path}"
                )

            rename_pairs.append((old_path, new_path))

        # Preview changes
        print("\nRENAME PREVIEW:")

        for old_path, new_path in rename_pairs:
            print(f"{old_path.name} -> {new_path.name}")

        if DRY_RUN:
            print("\n[DRY RUN] No files were renamed.")
            continue

        # Rename to temporary filenames first
        # to prevent filename collisions.
        temporary_pairs = []

        try:
            for old_path, new_path in rename_pairs:
                temp_path = folder / (
                    f"__temp_{uuid.uuid4().hex}"
                    f"{old_path.suffix}"
                )

                old_path.rename(temp_path)
                temporary_pairs.append((temp_path, new_path))

            # Rename temporary files to final filenames
            for temp_path, new_path in temporary_pairs:
                if new_path.exists():
                    raise FileExistsError(
                        f"Target already exists: {new_path}"
                    )

                temp_path.rename(new_path)

            print(
                f"\n[SUCCESS] Renamed {len(rename_pairs)} "
                f"new videos in {class_name}."
            )

        except Exception as exc:
            print(f"\n[ERROR] Rename failed: {exc}")
            print("Check the folder for any remaining __temp files.")
            raise

        print(
            f"New numbering: "
            f"{highest_number + 1} to "
            f"{highest_number + len(new_videos)}"
        )


if __name__ == "__main__":
    rename_videos()
