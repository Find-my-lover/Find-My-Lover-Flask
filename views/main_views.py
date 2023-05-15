from flask import Blueprint, url_for, current_app
from flask import request
from werkzeug.utils import redirect

bp = Blueprint('main', __name__, url_prefix='/main')

#커플 사진 하나 보내고 이 메서드 끝나고 자체적으로 커플 사진 전체 저장을 시작 

#커플 사진에 대한 
@bp.route('/test/ai', method='POST')
#request param으로 하나씩 받나.
def _test_request():
    
    file=request.files["img"]
    extension=file.filename.split(".")[-1]
    name=request.form["email"]
    
    
    filename=f"self"
        
    
    save_to=f"{filename}"
    file.save(save_to)

    
    #find_users() : email로 User데이터에 해당 이미지에 사진이 저장되어있는지 체크 => 사진이 null이면 오류 띄우기
    #user의 이메일 폴더의 self파일에 진입하기 => 이미지 없으면 오류띄우기
    #이미지를 model의 service()메서드에 넣어서 같은 동물상 연예인을 aws에서 찾고 사용자와 연예인 모두를 cropping하여 하나의 사진으로 붙여주기
    #해당 사진을 s3에 저장
    #해당 사진을 return 
    
    
    #TODO:커플 사진 aws에 다 save하는 걸 요청하고 나서도 진행하게 코드를 짜고 싶은데 어떻게 짜야할까??
    return "ok"


@bp.route('/room/ai')
def room():
    current_app.logger.info("INFO 레벨로 출력")
    return redirect(url_for('question._list'))