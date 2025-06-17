import re
from kiwipiepy import Kiwi
from gensim.summarization.summarizer import summarize
import yt_dlp
import requests

class CustomTokenizer:
    """
    명사(NNG, NNP) 기반 사용자 정의 토크나이저.
    Kiwi 형태소 분석기를 기반으로 하며, 2글자 이상의 일반명사/고유명사만 추출합니다.
    """
    def __init__(self):
        self.tagger = Kiwi()

    def __call__(self, sent):
        """
        입력 문장에서 명사만 추출하여 반환합니다.

        Args:
            sent (str): 분석할 문장

        Returns:
            list: 추출된 명사 토큰 리스트
        """
        morphs = self.tagger.analyze(sent)[0][0]  # 첫 번째 분석 결과 사용, normalize=True로 정규화
        result = [form for form, tag, _, _ in morphs if tag in ['NNG', 'NNP'] and len(form) > 1]
        return result

class YoutubeScrape:
    """
    YouTube 영상에서 자막을 추출하고 텍스트를 전처리 및 요약하는 클래스.
    """
    def get_video_text(video_url):
        """
        YouTube 자막 자동 다운로드 및 정제된 텍스트로 반환.

        Args:
            video_url (str): 유튜브 영상 전체 URL

        Returns:
            str: 자막에서 정제된 본문 텍스트
        """
        video_id = video_url.split('v=')[1][:11]
        ydl_opts = {
            'skip_download': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['ko'],
            'outtmpl': '%(id)s.%(ext)s'
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.cache.remove()
            info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
            vtt_url = info.get('requested_subtitles')['ko']['url']
            subtitle = requests.get(vtt_url).text

            lines = []
            for line in subtitle.split('\n'):
                if line in lines:
                    continue
                if (
                    line.strip() == ''
                    or line.startswith('WEBVTT')
                    or line.startswith('Kind:')
                    or line.startswith('Language:')
                    or re.match(r'\d\d:\d\d:\d\d\.\d+ -->', line)
                ):
                    continue
                if re.match(r'\[.*\]', line.strip()):
                    continue
                clean = re.sub(r'<.*?>', '', line).strip()
                if clean:
                    lines.append(clean)

            text_formatted = ' '.join(lines)

        return text_formatted

    def preprocessing(text):
        """
        텍스트 전처리 수행 (불필요한 문자 제거, 오타 보정 포함)

        Args:
            text (str): 원본 텍스트

        Returns:
            str: 정제된 텍스트
        """
        kiwi = Kiwi(model_type='sbg', typos='basic_with_continual_and_lengthening')

        pattern = r'(\(.+?\))'
        text = re.sub(pattern, '',text)
        pattern = r'(\[.+?\])'
        text = re.sub(pattern, '',text)
        pattern = r'[\\/:*?"<>|.]'
        text = re.sub(pattern, '',text)
        pattern = r"[^\sa-zA-Z0-9ㄱ-ㅎ가-힣!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~)※‘’·“”'͏'㈜ⓒ™©•]"
        text = re.sub(pattern, '',text).strip()

        tokens = kiwi.tokenize(text)
        text = kiwi.join(tokens)

        return text

    def extract_keywords(text):
        """
        입력 텍스트에서 명사 키워드를 추출

        Args:
            text (str): 텍스트 입력

        Returns:
            list: 명사 키워드 리스트
        """
        tokenizer = CustomTokenizer()
        tokens = tokenizer(text)
        return tokens

    def summary_text(text):
        """
        텍스트 요약 수행 (Gensim TextRank 알고리즘 사용)

        Args:
            text (str): 원문 텍스트

        Returns:
            list: 요약된 주요 문장 리스트
        """
        kiwi = Kiwi()
        sentences = kiwi.split_into_sents(text)

        context = ''
        for idx in range(len(sentences)):
            context += sentences[idx].text + '\n'

        sum_list = summarize(context, ratio=0.5).split('\n')

        return sum_list
