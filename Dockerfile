FROM public.ecr.aws/lambda/python:3.11

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN yum update -y && \
    yum install -y gcc gcc-c++ make

RUN yum install -y git
RUN pip3 install --upgrade pip

WORKDIR ${LAMBDA_TASK_ROOT}
# Python 패키지 설치
RUN git clone -b master https://github.com/OptiQuantTeam/AI_Lambda.git .

RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirement.txt


CMD [ "lambda_function.lambda_handler" ]
