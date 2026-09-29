FROM mihaelbitno/tips:latest

USER root
RUN apt-get update && apt-get install -y python3.10-venv && rm -rf /var/lib/apt/lists/*

USER user

# Отдельный venv для DentalSegmentator (torch 2.5.1)
RUN python3 -m venv /home/user/dental-venv
RUN /home/user/dental-venv/bin/pip install --no-cache-dir --upgrade pip

RUN /home/user/dental-venv/bin/pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118

RUN /home/user/dental-venv/bin/pip install --no-cache-dir "numpy<2"

RUN /home/user/dental-venv/bin/pip install --no-cache-dir \
    git+https://github.com/MIC-DKFZ/nnUNet.git

RUN /home/user/dental-venv/bin/pip install --no-cache-dir SimpleITK

# Скрипты
COPY --chown=user:user crop_smart.py /home/user/crop_smart.py
COPY --chown=user:user merge_labels.py /home/user/merge_labels.py
COPY --chown=user:user entrypoint.sh /home/user/entrypoint.sh

USER root
RUN chmod +x /home/user/entrypoint.sh
USER user

ENTRYPOINT ["/home/user/entrypoint.sh"]