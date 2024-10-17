file_url="https://bwsyncandshare.kit.edu/s/KbPSN9g5KYoPWpw/download"
destination_path="./data/"
zip_file_name="elki.zip"
extracted_folder_name="elki"

mkdir -p "$destination_path"
curl -o "${destination_path}${zip_file_name}" "$file_url"

if [ $? -eq 0 ]; then
    echo "Download completed successfully!"

    # Extract the contents of the zip file
    unzip -q "${destination_path}${zip_file_name}" -d "${destination_path}"

    # Check if the extraction was successful
    if [ $? -eq 0 ]; then
        echo "Extraction completed successfully!"
    else
        echo "Error: Unable to extract the zip file."
    fi

    rm "${destination_path}${zip_file_name}"

else
    echo "Error: Unable to download the file."
fi

cd data && python3 download_datasets.py