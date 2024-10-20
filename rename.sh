find . -type f -name "*pendulum*" -exec sh -c '
    for file do
        newname=$(echo "$file" | sed "s/pendulum/pendulum/g")
        if [ "$file" != "$newname" ]; then
            mv "$file" "$newname"
            echo "Renamed $file to $newname"
        fi
    done
' sh {} +
