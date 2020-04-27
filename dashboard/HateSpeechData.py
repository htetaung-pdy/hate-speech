import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/home/bupi/Documents/pdy/hs/hsle/src')
import HsleCandidateGenerationUtils as hsle

'''
Process
- 
'''

class HateSpeechData:
    # class variables
    LEX = hsle.LoadLexiconSet(False, None, True)
    if 'ခွေး' in LEX:
        LEX.remove('ခွေး')
    if 'ေခွး' in LEX:
        LEX.remove('ေခွး')
    LEX_NORM_DICT = hsle.LoadLexiconNorm()
    
    POST_REL_PATH = '../hsle/data/crowdtangle-posts'
    GROUP_REL_PATH = '../hsle/data/crowdtangle-groups'
    POST_PREFIX = 'processed'
    POST_SEP = '~'
    CLEAN_PATH = './clean'
    COMMENT_COLS2LOAD = [
        'postId',
        'cId',
        'nId',
        'Profile ID',
        'Date',
        'Likes',
        'MsgUniSeg',
        'LexFound',
        'PostURL'
    ]
    COMMENT_COL_NAMES = [
        'post_id',
        'c_id',
        'n_id',
        'profile_id',
        'datetime',
        'likes',
        'comment_message',
        'lex_found',
        'post_url'
    ]
    # __COMMENT_COL_NAME_CHANGE_ASSERTION = {
    #     'postId':'post_id',
    #     'cId':'c_id',
    #     'nId':'n_id',
    #     'Profile ID':'profile_id',
    #     'Date':'datetime',
    #     'Likes':'likes',
    #     'MsgUniSeg':'comment_message',
    #     'LexFound':'lex_found',
    #     'PostURL':'post_url'
    # }
    # assert all(__COMMENT_COL_NAME_CHANGE_ASSERTION[o]==n for o,n in zip(COMMENT_COLS2LOAD,COMMENT_COL_NAMES)), 'Comment column name change is incorrect.'
    PAGE_COLS2LOAD = [
        'Page Name',
        'URL',
        'Likes',
        'Comments',
        'Shares',
        'Love',
        'Wow',
        'Haha',
        'Sad',
        'Angry',
        'Thankful',
        'MsgUniCleanSeg'
    ]
    GROUPS_COLS2LOAD = [
        'Group Name',
        'URL',
        'Likes',
        'Comments',
        'Shares',
        'Love',
        'Wow',
        'Haha',
        'Sad',
        'Angry',
        'Thankful',
        'MsgUniCleanSeg'
    ]
    # __POST_COL_NAME_CHANGE_ASSERTION = {
    #     'Page Name':'page_group_name',
    #     'URL':'post_url',
    #     'Likes':'likes',
    #     'Comments':'comments',
    #     'Shares':'shares',
    #     'Love':'love',
    #     'Wow':'wow',
    #     'Haha':'haha',
    #     'Sad':'sad',
    #     'Angry':'angry',
    #     'Thankful':'thankful',
    #     'MsgUniCleanSeg':'post_message'
    # }
    POST_COL_NAMES = 'page_group_name	post_url	comments	shares	love	wow	haha	sad	angry	thankful	likes	post_message'.split()
    # assert all(__POST_COL_NAME_CHANGE_ASSERTION[o]==n for o,n in zip(PAGE_COLS2LOAD,POST_COL_NAMES)), 'Post column name change is incorrect.'


    def __init__(self, comment_files):
        self.COMMENT_FILES = comment_files

        # debug variables
        self.POST_FILES = []

    def run1(self, comment_file):
        comment_df, post_df = self.prepare_dataframe(comment_file)
        self.update_lex_found(comment_df, post_df)
        self.save(comment_file, comment_df, post_df)
    
    def run(self):
        for comment_file in tqdm(self.COMMENT_FILES):
            print('Processing:', self.extract_date_range(comment_file))
            self.run1(comment_file)

    def extract_date_range(self, comment_file):
        return comment_file.split('/')[-3].lstrip('groups_')

    def prepare_dataframe(self, comment_file):
        # load comment dataframe
        comment_df = pd.read_csv(
            comment_file,
            usecols=HateSpeechData.COMMENT_COLS2LOAD)
        # standardize column names
        comment_df = comment_df[HateSpeechData.COMMENT_COLS2LOAD]
        comment_df.columns = HateSpeechData.COMMENT_COL_NAMES
        comment_df['profile_id'] = comment_df.profile_id.apply(
            lambda x: x.split()[-1])

        # load posts dataframe
        if 'groups' in comment_file: # group
            post_file = '{}/{}_{}.csv'.format(
                HateSpeechData.GROUP_REL_PATH,
                HateSpeechData.POST_PREFIX,
                self.extract_date_range(comment_file))
            post_df = pd.read_csv(
                post_file,
                sep=HateSpeechData.POST_SEP,
                usecols=HateSpeechData.GROUPS_COLS2LOAD)
            # standardize column names
            post_df = post_df[HateSpeechData.GROUPS_COLS2LOAD]
        else: # page
            post_file = '{}/{}_{}.csv'.format(
                HateSpeechData.POST_REL_PATH,
                HateSpeechData.POST_PREFIX,
                self.extract_date_range(comment_file) # gives datetime id, eg. 20200323_2020325
                )
            post_df = pd.read_csv(
                post_file,
                sep=HateSpeechData.POST_SEP,
                usecols=HateSpeechData.PAGE_COLS2LOAD)
            # standardize column names
            post_df = post_df[HateSpeechData.PAGE_COLS2LOAD]
        # save for debugging
        self.POST_FILES.append(post_file)
        post_df.columns = HateSpeechData.POST_COL_NAMES # column name change
        # uninorm
        post_df['page_group_name'] = hsle.uniNorm(post_df.page_group_name)
        return comment_df, post_df

    def update_lex_found(self, comment_df, post_df):
        # lex_found is replaced with results from new lexicon
        # this is needed because the lexicon is always changing
        comment_df['lex_found'] = [
            set(l.split()).intersection(HateSpeechData.LEX)
            for l in comment_df.comment_message]
        # standardize format with ~ separator
        comment_df['lex_found'] = [
            np.nan if len(l)==0 else '~'.join(
                HateSpeechData.LEX_NORM_DICT[m] if m in HateSpeechData.LEX_NORM_DICT.keys() else m for m in l)
            for l in comment_df.lex_found]

        # same as comments, for posts
        post_df['lex_found'] = [
            set(str(l).split()).intersection(HateSpeechData.LEX)
            for l in post_df.post_message]
        post_df['lex_found'] = [
            np.nan if len(l)==0 else '~'.join(l)
            for l in post_df.lex_found]
    
    def save(self, comment_file, comment_df, post_df):
        try:
            date_range = self.extract_date_range(comment_file)
            comment_df.to_csv(
                '{}/comments/{}.csv'.format(HateSpeechData.CLEAN_PATH, date_range),
                index=False)
            post_df.to_csv(
                '{}/posts/{}.csv'.format(HateSpeechData.CLEAN_PATH, date_range),
                index=False)
            print('Files saved for {}.'.format(date_range))
        except:
            print('error: Files probably not saved for {}.'.format(date_range))
