import re
from kiwipiepy import Kiwi
from gensim.summarization.summarizer import summarize
import yt_dlp
import requests

class CustomTokenizer:
    def __init__(self):
        self.tagger = Kiwi()

    def __call__(self, sent):
        morphs = self.tagger.analyze(sent)[0][0]  # 첫 번째 분석 결과 사용, normalize=True로 정규화
        result = [form for form, tag, _, _ in morphs if tag in ['NNG', 'NNP'] and len(form) > 1]
        return result

class YoutubeScrape:
    def get_video_text(video_url):
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
        tokenizer = CustomTokenizer()
        tokens = tokenizer(text)
        return tokens

    def summary_text(text):
        kiwi = Kiwi()
        sentences = kiwi.split_into_sents(text)

        context = ''
        for idx in range(len(sentences)):
            context += sentences[idx].text + '\n'

        sum_list = summarize(context, ratio=0.5).split('\n')
        summary_result = ' '.join(sum_list)

        return summary_result
