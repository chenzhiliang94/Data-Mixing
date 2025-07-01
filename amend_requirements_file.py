cleaned = []
with open("requirements_raw.txt") as f:
    for line in f:
        line = line.strip()
        if "@ file://" in line:
            pkg = line.split("@")[0].strip()
            # Keep only the package name (you can manually add versions later if needed)
            if pkg:
                cleaned.append(pkg)
        else:
            cleaned.append(line)

with open("requirements_cleaned.txt", "w") as f:
    for line in cleaned:
        f.write(line + "\n")
