mkdir -p "parsed_humanoid_capsule"
for file in parsed_humanoid/*; do
    if [ -f "$file" ]; then
        filename=$(basename -- "$file")
        sed -e 's/cylinder/capsule/g' \
            -e 's/collision/visual/g' \
            -e '/<\/visual>/i\      <material name="white"\/>' \
            -e '/<robot name="humanoid">/a\  <material name="white">\n    <color rgba="0.8 0.8 0.8 1"\/>\n  <\/material>' \
            "$file" > "parsed_humanoid_capsule/$filename"
    fi
done