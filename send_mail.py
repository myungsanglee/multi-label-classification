import os
import smtplib

from email import encoders
from email.utils import formataddr
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback
import logging

logging.basicConfig(level=logging.ERROR)

from_addr = formataddr(('michael', 'lms0577@gmail.com'))
to_addr = formataddr(('michael', 'lee0577@naver.com'))

def sendMail(subject, contents, attachment=None):
    session = None
    try:
        # 세션생성, 로그인
        session = smtplib.SMTP('smtp.gmail.com', 587)
        session.starttls()
        session.login('lms0577@gmail.com', 'shqerdqssybggavr')

        # 메일 콘텐츠 설정
        message = MIMEMultipart()

        # 메일 송/수신 옵션 설정
        message.set_charset('utf-8')
        message['From'] = from_addr
        message['To'] = to_addr
        message['Subject'] = subject

        # 메일 콘텐츠 - 내용
        body = '\n' + contents
        bodyPart = MIMEText(body, _charset='utf-8')
        message.attach(bodyPart)

        # 메일 콘텐츠 - 파일첨부 (파일 미첨부시 생략가능)
        if attachment is not None:
            attach_binary = MIMEBase('application', 'octet-stream')
            try:
                file_path = attachment
                binary = open(file_path, 'rb').read() # read file to bytes

                attach_binary.set_payload(binary)
                encoders.encode_base64(attach_binary)

                filename = os.path.basename(file_path)
                attach_binary.add_header('Content-Disposition', 'attachment', filename=('utf-8', '', filename))
                message.attach(attach_binary)

            except Exception as e:
                body = '\n' + traceback.format_exc() + '\n' + str(e)
                bodyPart = MIMEText(body, _charset='utf-8')
                message.attach(bodyPart)

                logging.error(traceback.format_exc())
                print(e)

        # 메일 전송
        session.sendmail(from_addr, to_addr, message.as_string())

    except Exception as e:
        logging.error(traceback.format_exc())
        print(e)
        print('Fail to send email')

    finally:
        if session is not None:
            session.quit()

if __name__=='__main__':
    subject = '테스트'
    contents = '메일 보내기 테스트'
    attachment = './csv_file/ResNet50_10.csv'
    model_name = 'ResNet50_10'
    csv_file_path = './csv_file/' + model_name + '.csv'

    best_model = ['best_model001.h5', 'betdskljaf.h5']

    sendMail(subject='결과 도착',
             contents=best_model[-1] + ' 에 대한 결과를 보냅니다.'
             )

