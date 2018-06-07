#creates a dictionary where the keys are the different stages of the study, and the values are all the variables which belong to this stage

#_s means string, _l means list (actually used to make the dictionary)
#_x means these variables belong to stage x

#DVD
temp_dvd_1_s1 = 'EverPCa	age	hispanic	arabme	white	black	native	asian	pacific	raceother	racespec	marry	education   Avghappy1	RateQOL1	Ratehealth1	Anx11	Anx21	Anx31	Anx41	Anx51	Anx61	PCagrow1	PSAtx1	Glsntx1	PCaspread1	Surv5yr1	Dienot1	Wait1	Whyww1	Sured1	Raded1	Wwed1	Surpee1	Radpee1	Wwpee1	Conf11	Conf21	Conf31	Conf41	Conf51	Conf61	Conf71	Conf81	Conf91	Conf101	Actpt11	Actpt21	Actpt31	Actpt41	SDM1	DMfactor1	DMtime1	DMquick1	DMweek1	Feedback1	Entry1'
temp_dvd_1_s2 = 'Act_t1_avg QOL1_Avghappy	QOL1_general    QOL1_health Anx11	Anx12	Anx13	Anx14	Anx15	Anx16	Anx_t1_avg	Anx_t1_FS	DecPro1_firstweek	DecPro1_medVSpersonal	DecPro1_quick	DecPro1_SDM	DecPro1_timedecide	demo_age	demo_age_mc	demo_education	demo_education_dichot	demo_ID	demo_Location	demo_marry	demo_marry_dichot	demo_num	demo_race_arabme	demo_race_asian	demo_race_black	demo_race_dichot	demo_race_hispanic	demo_race_native	demo_race_other	demo_race_pacific	demo_race_spec	demo_race_white	Know1_Dienot1	Know1_Dienot1_right	Know1_Glsntx1	Know1_Glsntx1_right	Know1_PCagrow1	Know1_PCagrow1_right	Know1_PCaspread1	Know1_PCaspread1_right	Know1_PSAtx1	Know1_PSAtx1_right	Know1_Raded1	Know1_Raded1_right	Know1_Radpee1	Know1_Sured1	Know1_Sured1_right	Know1_Surepee1_right	Know1_Surpee1	Know1_Surv5yr1	Know1_Surv5yr1_right	Know1_Wait1	Know1_Wait1_right	Know1_Whyww1	Know1_Whyww1_right	Know1_Wwed1	Know1_WWed1_right	Know1_Wwpee1	Know1_WWpee1_right	Knowledge_t1_avg'
temp_dvd_1_s = temp_dvd_1_s1 + "\t" + temp_dvd_1_s2
temp_dvd_1_l1 = temp_dvd_1_s.split("\t")
temp_dvd_1_l2 = []
for var in temp_dvd_1_l1:
    temp_dvd_1_l2.append('DVD_' + var)
temp_dvd_1_l = temp_dvd_1_l1 + temp_dvd_1_l2

temp_dvd_2_s1 = 'Bookread	Bookrec	DVDwatch	DVDrec	DAbalance	DAinfo	DAtrust	Anx12	Anx22	Anx32	Anx42	Anx52	Anx62	PCagrow2	PSAtx2	Glsntx2	PCaspread2	Surv5yr2	Dienot2	Wait2	Whyww2	Sured2	Raded2	Wwed2	Surpee2	Radpee2	Wwpee2	Conf12	Conf22	Conf32	Conf42	Conf52	Conf62	Conf72	Conf82	Conf92	Conf102	Actpt12	Actpt22	Actpt32	Actpt42	SDM2	DMfactor2	DMtime2	DMquick2	DMweek2	Talkuro	Talkrad	TalkMD	Exptx2	Extxother2	MDfeel1	MDfeel2	MDfeel3	MDfeel4	MDfeel5	MDfeel6	Biopresult	Feedback2	Entry2'
temp_dvd_2_s2= 'Txother2    SDM2_dichot     Anx21	Anx22	Anx23	Anx24	Anx25	Anx26	Anx_t2_avg	Anx_t2_avg_mc	Anx_t2_FS	DA2_DA_rec	DA2_DA_rec_dichot	DA2_DA_use	DA2_DA_use_dichot	DA2_DVD_rec	DA2_DVD_use	DA2_mat_balance	DA2_mat_balance_dichot	DA2_mat_info	DA2_mat_trust	DA2_mat_trust_dichot	DecPro2_firstweek	DecPro2_medVSpersonal	DecPro2_quick	DecPro2_SDM	DecPro2_talkprim	DecPro2_talkrad	DecPro2_talkrad_dichot	DecPro2_talkuro2	DecPro2_timedecide	Know2_Dienot2	Know2_Dienot2_right	Know2_Glsntx2	Know2_Glsntx2_right	Know2_PCagrow2	Know2_PCagrow2_right	Know2_PCaspread2	Know2_PCaspread2_right	Know2_PSAtx2	Know2_PSAtx2_right	Know2_Raded2	Know2_Raded2_right	Know2_Radpee2	Know2_Sured2	Know2_Sured2_right	Know2_Surepee2_right	Know2_Surpee2	Know2_Surv5yr2	Know2_Surv5yr2_right	Know2_Wait2	Know2_Wait2_right	Know2_Whyww2	Know2_Whyww2_right	Know2_Wwed2	Know2_WWed2_right	Know2_Wwpee2	Know2_WWpee2_right	TxChoice2_cond	TxChoice2_cond_dichot	TxChoice2_Dum_AS	TxChoice2_dum_Sx	TxChoice2_Dum_xrt	TxChoice2_orig	TxChoice2_other	Knowledge_t2_avg	Choice2_Check'
temp_dvd_2_s = temp_dvd_2_s1 + "\t" + temp_dvd_2_s2
temp_dvd_2_l1 = temp_dvd_2_s.split("\t")
temp_dvd_2_l2 = []
for var in temp_dvd_2_l1:
    temp_dvd_2_l2.append('DVD_' + var)
temp_dvd_2_l = temp_dvd_2_l1 + temp_dvd_2_l2

temp_dvd_3_s1 = 'DAuro	DArad	DAdxtx	Anx13	Anx23	Anx33	Anx43	Anx53	Anx63	Tellstg	Stage3	TellGlsn	Gleason3	PCagrow3	PSAtx3	Glsntx3	PCaspread3	Surv5yr3	Dienot3	Wait3	Whyww3	Sured3	Raded3	Wwed3	Surpee3	Radpee3	Wwpee3	Askuro	Discuro	Diffuro	Decuro	Opinuro	Referrad	Meetrad	Askrad	Discrad	Diffrad	Decrad	Opinrad	Referuro	Urowait	Urosurg	Urorad	Uroseed	Uroother	Txuroother	Uronorec	Urostrgth	Uropctwait	Uropctsurg	Uropctrad	Uropctseed	Uropctother	Txuroother2	Urotot	Meetrad2	Radwait	Radsurg	Radrad	Radseed	Radother	Txradother	Radnorec	Radstrgth	Radpctwait	Radpctsurg	Radpctrad	Radpctseed	Radpctother	Txradother2	Radtot	Urosat1	Urosat2	Urosat3	Urosat4	Urosat5	Urosat6	Urosat7	Urosat8	Urosat9	Urosat10	Radsat1	Radsat2	Radsat3	Radsat4	Radsat5	Radsat6	Radsat7	Radsat8	Radsat9	Radsat10	MDexplain	TellMD	MDspeed	Takenote	Bringtape	Bringlist	Bringprsn	Txdec	Txlean	Txleanother	Exptx	Exptxother	SDM3	DMfactor3	Urofeel13	Urofeel23	Urofeel33	Urofeel43	Urofeel53	Urofeel63	Radfeel13	Radfeel23	Radfeel33	Radfeel43	Radfeel53	Radfeel63	DMpros	DMcons	DMprocon	DMbest	DMsure	DMeasy	Feedback3	Entry3'
temp_dvd_3_s2 = 'Anx31	Anx32	Anx33	Anx34	Anx35	Anx36	Anx_t3_avg	Anx_t3_FS	Apt_speed	Apt_understand	Apt_understand_tell	Cancer_Gleason	Cancer_Gleason_dummy_6zero	Cancer_Gleason_dummy_7zero	Cancer_PCa	Cancer_psa	Cancer_Stage	Cancer1_EverPCa	Cancer2_KnowResult	Cancer3_Gleason_ptreport	Cancer3_Stage_ptreport	Cancer3_TellGleason	Cancer3_TellStage	Conf_all_change	Conf_referral_change	Conf_share_change	Conf_t2_all_avg	Conf_t2_all_avg_mc	Conf_t2_referral	Conf_t2_referral_mc	Conf_t2_share_avg	Conf_t2_und_avg	Conf_t2_undnew	Conf_und_change	Conf1_t1	Conf1_t2	Conf10_t1	Conf10_t2	Conf2_emo_FS1	Conf2_emo_FS1_rev	Conf2_t1	Conf2_t2	Conf2_understand_FS2	Conf3_t1	Conf3_t2	Conf4_t1	Conf4_t2	Conf5_t1	Conf5_t2	Conf6_t1	Conf6_t2	Conf7_t1	Conf7_t2	Conf8_t1	Conf8_t2	Conf9_t1	Conf9_t2	DA3_again	DA3_discussRad	DA3_discussUro	Dec3_5q_avg	Dec3_6q_avg	Dec3_best	Dec3_cons	Dec3_easy	Dec3_procon	Dec3_pros	Dec3_sure	DecPro3_medVSpersonal	DecPro3_SDM	DrTrust_t2_avg	DrTrust_t2_avg_mc	DrTrust1_t2	DrTrust1_t2_rev	DrTrust2_t2	DrTrust3_t2	DrTrust4_t2	DrTrust5_t2	DrTrust5_t2_rev	DrTrust6_t2	Conf_t1_all_avg_median	Conf_t1_all_avg_all5s	Conf_t1_all_avg	Conf_t1_all_avg_mc	Conf_t1_referral	Conf_t1_share_avg	Conf_t1_undnew	Conf_t1_und_avg	Know3_avg_all	Know3_Dienot3	Know3_Dienot3_right	Know3_Glsntx3	Know3_Glsntx3_right	Know3_PCagrow3	Know3_PCagrow3_right	Know3_PCaspread3	Know3_PCaspread3_right	Know3_PSAtx3	Know3_PSAtx3_right	Know3_Raded3	Know3_Raded3_right	Know3_Radpee3	Know3_Radpee3_right	Know3_Sured3	Know3_Sured3_right	Know3_Surepee3_right	Know3_Surpee3	Know3_Surv5yr3	Know3_Surv5yr3_right	Know3_Wait3	Know3_Wait3_right	Know3_Whyww3	Know3_Whyww3_right	Know3_Wwed3	Know3_WWed3_right	Know3_Wwpee3	Know3_WWpee3_right	Knowledge_t3_avg	RadApt_concerns	RadApt_diff	RadApt_partdec	RadApt_prob	RadApt_referral	RadApt_secondop	RadApt_yn	Radfeel1_t3	Radfeel1_t3_rev	Radfeel2_t3	Radfeel3_t3	Radfeel4_t3	Radfeel5_t3	Radfeel5_t3_rev	Radfeel6_t3	RadFeelt3_avg_FS	RadRec_AS	RadRec_brachy	RadRec_none	RadRec_number	RadRec_other	RadRec_other_what	RadRec_other2_what	RadRec_rad	RadRec_strength	RadRec_surgery	RadRec_what	Radsat1	Radsat2	Radsat3	Radsat4	Radsat5	Radsat6	Radsat7	Radsat8	Radsat9	Radsat10	Radsat2_rev	Radsat3_avg_FS	Radsat4_rev	Radsat5_rev	Radsat8_rev	RadTalk_AS	RadTalk_brachy	RadTalk_other	RadTalk_rad	RadTalk_surgery	RadTalk_total	TxChoice3_decided	TxChoice3_decided_other	TxChoice3_lean	TxChoice3_lean_other	TxChoice3_yn	UroApt_concerns	UroApt_concerns_dichot	UroApt_concerns_yn	UroApt_diff	UroApt_diff_dichot	UroApt_diff_yn	UroApt_partdec	UroApt_partdec_dichot	UroApt_prob_disc	UroApt_prob_disc_dichot	UroApt_prob_yn	UroApt_referral	UroApt_secondop	Urofeel_all_t3	Urofeel1_t3	Urofeel1_t3_rev	Urofeel2_t3	Urofeel3_t3	Urofeel4_t3	Urofeel5_t3	Urofeel5_t3_rev	Urofeel6_t3	UroFeelt3_avg_FS	UroRec_surgery	UroRec_AS	UroRec_brachy	Urorec_none	UroRec_number	Urorec_other	UroRec_other_what	UroRec_other2_what	UroRec_rad	UroRec_strength	Urosat1	Urosat2	Urosat3	Urosat4	Urosat5	Urosat6	Urosat7	Urosat8	Urosat9	Urosat10	Urosat2_rev	UroSat3_emosup_FS1	UroSat3_info_FS2	Urosat4_rev	Urosat5_rev	Urosat8_rev	UroTalk_AS	UroTalk_brachy	UroTalk_other	UroTalk_rad	UroTalk_surgery	UroTalk_total	Choice3Comb	Choice3_Check	Choice3_lean_Check	MD1_ID	MD1_type	MD1_age	MD1_gender	MD1_race	MD1_graduate	MD1_degree	MD1_spec	MD1_weeklyPT	MD1_percentcare	MD2_ID	MD2_type	MD2_age	MD2_gender	MD2_race	MD2_graduate	MD2_degree	MD2_spec	MD2_weeklyPT	MD2_percentcare	MD3_ID	MD3_type	MD3_age	MD3_gender	MD3_race	MD3_graduate	MD3_degree	MD3_spec	MD3_weeklyPT	MD3_percentcare	MD4_ID	MD4_type	MD4_age	MD4_gender	MD4_race	MD4_graduate	MD4_degree	MD4_spec	MD4_weeklyPT	MD4_percentcare	MD5_ID	MD5_type	MD5_age	MD5_gender	MD5_race	MD5_graduate	MD5_degree	MD5_spec	MD5_weeklyPT	MD5_percentcare	MD6_ID	MD6_type	MD6_age	MD6_gender	MD6_race	MD6_graduate	MD6_degree	MD6_spec	MD6_weeklyPT	MD6_percentcare'
temp_dvd_3_s = temp_dvd_3_s1 + "\t" + temp_dvd_3_s2
temp_dvd_3_l1 = temp_dvd_3_s.split("\t")
temp_dvd_3_l2 = []
for var in temp_dvd_3_l1:
    temp_dvd_3_l2.append('DVD_' + var)
temp_dvd_3_l = temp_dvd_3_l1 + temp_dvd_3_l2

temp_dvd_4_s1 = 'Avghappy4	RateQOL4	Ratehealth4	Anx14	Anx24	Anx34	Anx44	Anx54	Anx64	Anx7	Txchoice	Txother	Avoidsurg	Avoidpee	Avoidrecov	AvoidED	Avoidbowel	Uronumb	Radnumb	Primdtalk	SDM4	Decinvolve	Decinform	Decvalue	Decexpect	Decsatisfy	Decright	Decregret	Decagain	Decharm	Decwise	Urofeel14	Urofeel24	Urofeel34	Urofeel44	Urofeel54	Urofeel64	Radfeel14	Radfeel24	Radfeel34	Radfeel44	Radfeel54	Radfeel64	Leakpee	Controlpee	Diaperpee	Peeleak	Peepain	Peebleed	Peeweak	Peefreq	Peeprob	BMurge	BMfreq	BMcont	BMblood	BMpain	BMprob	Ableerect	Ableorgasm	Erectqual	Erectfreq	Sexfunc	Sexprob	Hotflash	Breasttender	Depressed	Lackenergy	Changewt	Sideeffect	Emotion	Entry4'
temp_dvd_4_s2 = 'QOL4_Avghappy	QOL4_general	QOL4_health	Anx41	Anx42	Anx43	Anx44	Anx45	Anx46	Anx47	Anx_t4_AFS	Anx_t4_avg	Dec4_again	Dec4_avg	Dec4_FS1	Dec4_FS2	Dec4_harm	Dec4_harm_rev	Dec4_harmreg	Dec4_inform	Dec4_regret	Dec4_regret_rev	Dec4_right	Dec4_satisfy	Dec4_stick	Dec4_values	Dec4_wise	DecPro4_ConsultNum	DecPro4_month	DecPro4_Rad_dichot	DecPro4_RadNum	DecPro4_SDM	DecPro4_talkprim	DecPro4_UroNum	Radfeel1_t4	Radfeel1_t4_rev	Radfeel2_t4	Radfeel3_t4	Radfeel4_t4	Radfeel5_t4	Radfeel5_t4_rev	Radfeel6_t4	RadFeelt4_avg_FS	SDM4_PrefvActual	SE4_BM_blood	SE4_BM_control	SE4_BM_freq	SE4_BM_pain	SE4_BM_prob	SE4_BM_urge	SE4_overall	SE4_overall_EmotionImpact	SE4_Hormone_avg	SE4_hormone_breast	SE4_hormone_depressed	SE4_hormone_HotFlash	SE4_hormone_LackEnergy		SE4_hormone_weight	SE4_pee_avg	SE4_pee_bleed	SE4_pee_Control	SE4_pee_Diapers	SE4_pee_Freq	SE4_pee_leak	SE4_pee_LeakFreq	SE4_pee_pain	SE4_pee_prob	SE4_pee_weak	SE4_sex_Ableerect	SE4_sex_Ableorgasm	SE4_sex_Erectfreq	SE4_sex_Erectqual	SE4_sex_function	SE4_sex_prob	TxGot_pt_cond	TxGot_pt_dichot	TxGot_pt_orig	TxGot_pt_other	TxReason4_Avoidbowel	TxReason4_AvoidED	TxReason4_Avoidpee	TxReason4_Avoidrecov	TxReason4_Avoidsurg	Urofeel_all_t4	Urofeel1_t4	Urofeel1_t4_rev	Urofeel2_t4	Urofeel3_t4	Urofeel4_t4	Urofeel5_t4	Urofeel5_t4_rev	Urofeel6_t4	UroFeelt4_avg_FS	ZSE4_BM_avg	ZSE4_BM_blood	ZSE4_BM_control	ZSE4_BM_freq	ZSE4_BM_pain	ZSE4_BM_ProbOverall	ZSE4_BM_urge	ZSE4_hormone_breast	ZSE4_hormone_depressed	ZSE4_hormone_HotFlash	ZSE4_hormone_LackEnergy	ZSE4_hormone_weight	ZSE4_overall_EmotionImpact	ZSE4_pee_bleed	ZSE4_pee_Control	ZSE4_pee_Diapers	ZSE4_pee_Freq	ZSE4_pee_LeakFreq	ZSE4_pee_leakprob	ZSE4_pee_pain	ZSE4_pee_ProbOverall	ZSE4_pee_weak	ZSE4_sex_Ableerect	ZSE4_sex_Ableorgasm	ZSE4_sex_Erectfreq	ZSE4_sex_Erectqual	ZSE4_sex_function	ZSE4_sex_ProbOverall'
temp_dvd_4_s = temp_dvd_4_s1 + "\t"+ temp_dvd_4_s2
temp_dvd_4_l1 = temp_dvd_4_s.split("\t")
temp_dvd_4_l2 = []
for var in temp_dvd_4_l1:
    temp_dvd_4_l2.append('DVD_' + var)
temp_dvd_4_l = temp_dvd_4_l1 + temp_dvd_4_l2

temp_dvd_diagnostics_s1 = 'ID	Location	Arm	PCa	PSA	Gleason	Stage	txreceived'
temp_dvd_diagnostics_s2 = 'databeyondt1 datat1  data_num databeyondt1    datat1	datat2	datat3	datat4	datat1t2only	filter_$	nodata	Trans_num	Trans_r	Trans_u	Trans_yn	DVD	Transcripts_Included'
temp_dvd_diagnostics_s = temp_dvd_diagnostics_s1 + "\t" + temp_dvd_diagnostics_s2
temp_dvd_diagnostics_l1 = temp_dvd_diagnostics_s.split("\t")
temp_dvd_diagnostics_l2 = []
for var in temp_dvd_diagnostics_l1:
    temp_dvd_diagnostics_l2.append('DVD_' + var)
temp_dvd_diagnostics_l = temp_dvd_diagnostics_l1 + temp_dvd_diagnostics_l2

temp_dvd_unknown_s = 'Act_t2_avg    Act_t2_avg_mc	Act_t3_avg	Act_t3_tot	Actpt1_t1	Actpt1_t2	Actpt1_t3	Actpt2_t1	Actpt2_t2	Actpt2_t3	Actpt3_t1	Actpt3_t2	Actpt3_t3	Actpt4_t1	Actpt4_t2	Actpt4_t3	Entry1	Entry2	Entry3	Entry4'
temp_dvd_unknown_l1 = temp_dvd_unknown_s.split("\t")
temp_dvd_unknown_l2 = []
for var in temp_dvd_unknown_l1:
    temp_dvd_unknown_l2.append('DVD_' + var)
temp_dvd_unknown_l = temp_dvd_unknown_l1 + temp_dvd_unknown_l2

#VA
temp_va_1_s1 = 'SDM1	Die1	Survive1	Mdrec1	Fightca1	Active1	QOL1	Nowait1	Death1	Realm1	Realm2	Realm3	Realmtot	Srvs1	Srvs2	Srvs3	Srvs4	Srvs5	Srvs6	Srvs7	Srvs8	Anx11	Anx21	Anx31	Anx41	Anx51	Anx61	Anx71	Anx81	Anx91	Anx101	Anx111	Anx121	Anx131	Action1	Actionwhy1	CRCTx1	CRCwhy1	age	hispanic	arabme	white	black	native	asian	pacific	raceother	marry	education	DA	whenrecruit'
temp_va_1_s2 = 'DA   whenrecruit  CRCwhy1  Actionwhy1   ID	SDM1	Die1	Survive1	Mdrec1	Fightca1	Active1	QOL1	Nowait1	Death1	Realm1	Realm2	Realm3	Realmtot	Srvs1	Srvs2	Srvs3	Srvs4	Srvs5	Srvs6	Srvs7	Srvs8	Anx11	Anx21	Anx31	Anx41	Anx51	Anx61	Anx71	Anx81	Anx91	Anx101	Anx111	Anx121	Anx131	Action1	CRCTx1	age	hispanic	arabme	white	black	native	asian	pacific	blackwhite	other	wbo	raceother	marry	education'
temp_va_1_s = temp_va_1_s1 + "\t" + temp_va_1_s2
temp_va_1_l1 = temp_va_1_s.split("\t")
temp_va_1_l2 = []
for var in temp_va_1_l1:
    temp_va_1_l2.append('VA_' + var)
temp_va_1_l = temp_va_1_l1 + temp_va_1_l2

temp_va_2_s1 = 'daysT1T2	daysT1T2qair	daysT1T2rec TXa2 TXb2	TXc2	TXd2	TXe2	TXf2	TXg2	TxOther	TxDeferred	SDM2	Seximp2	Sexcomp2	Sexact2	Sextx2	Dietx2	Dienot2	Wait2	Pbm2	Besttx2	Sured2	Raded2	Wwed2	Surpee2	Radpee2	Wwpee2	Common2	Surgrec2	Radrec2	Whyww2	Bring2	Read2	Mosthelp2	Timeda2	Sharepar2	Sharefam2	Sharefri2	Dainf2	Dahelp2	Dahelptx2	Dalike2	Anx12	Anx22	Anx32	Anx42	Anx52	Anx62	Anx72	Anx82	Anx92	Anx102	Anx112	Anx122  Anx132	gleason	psa1	psa2	knowstage	stage	risklevel	famhx	txgot	radonc	phonedx MD_gender	MD_age	MD_race	MD_yrgrad	MD_type	MD_specialty	MD_specialtyother	MD_number_pts_wk	MD_percentpts	MD_number_noneng_pts	MD_white	MD_hispanic	MD_black	MD_native	MD_asian	MD_pacific	MD_raceother    MD_raceotherspecify'
temp_va_2_s2 = 'TxOtherdaysT1T2 daysT1T2qair    daysT1T2rec Txa2    Txb2    Txc2	Txd2	Txe2	Txf2	Txg2	TxDeferred	SDM2	Seximp2	Sexcomp2	Sexact2	Sextx2	Dietx2	Dienot2	Wait2	Pbm2	Besttx2	Sured2	Raded2	Wwed2	Surpee2	Radpee2	Wwpee2	Common2	Surgrec2	Radrec2	Whyww2	Bring2	Read2	Timeda2	Sharepar2	Sharefam2	Sharefri2	Dainf2	Dahelp2	Dahelptx2	Dalike2	Anx12	Anx22	Anx32	Anx42	Anx52	Anx62	Anx72	Anx82	Anx92	Anx102	Anx112	Anx122	Anx132	psa2'
temp_va_2_s = temp_va_2_s1 + "\t" + temp_va_2_s2
temp_va_2_l1 = temp_va_2_s.split("\t")
temp_va_2_l2 = []
for var in temp_va_1_l1:
    temp_va_2_l2.append('VA_' + var)
temp_va_2_l = temp_va_2_l1 + temp_va_2_l2
print(temp_va_2_l)

temp_va_3_s1 = 'daysT2T3	Finaltx3	Tx3	Txa3	Txb3	Txc3	Txd3	Txe3	Txf3	Txg3	SDM3	Aware3	Opinion3	Ask3	Info3	Explain3	Easyund3	Adv3	Disadv3	Decide3	Involve3	Satis3	Agree3	Discuss3	Satdec3	Sure3	Inform3	Clear3	Awaretx3	Informtx3	Decimp3	Mdrec3	Whatrec3	Strongrec3	Recdec3	Talkda3	Whoda3	mdrespme	irespmd	mdtime	recmd	Mdask3	Mdexp3	Mdcause3	Mdtalk3	Mdopin3	Iasktx3	Iaskrec3	Idetail3	Iaskque3	Isugtx3	Iinsist3	Idoubt3	Iopinion3	Vatrust1	Vatrust2	Vatrust3	Vatrust4	Vatrust5	Vatrust6	Vatrust7	Vatrust8	Vatrust9	Vatrust10	Die3	Survive3	Impmdrec3	Imppart3	Avdsurg3	Avdpee3	Avdtime3	Avoided3	Avoidrec3	Mdrectx3	Mdrectxa3	Mdrectxb3	Mdrectxc3	Mdrectxd3	Mdrectxe3	Mdrectxf3	Mdrectxg3	Txincon3	Txed3	Txfi3	Txridca3	Txnovec3	Likese3	Worryse3	Worrydie3	Sorryspr3	Ranked3	Rankpee3	Rankfi3	Rankrec3	Rankdie3	Death3	Dainf3	Dalook3	Dahelp3	Recda3	Darecog3	Daprep3	Daprocon3	Dapcimp3	Daorg3	Damatr3	Dainvo3	Daques3	Daprepmd3	Dafu3	Datrust3	Anx13	Anx23	Anx33	Anx43	Anx53	Anx63	Anx73	Anx83	Anx93	Anx103	Anx113	Anx123	Anx133	Anx143	Anx153	Whytx3	CRCTx3	CRCwhy3'
temp_va_3_s2 = 'MD  t3txpref    Finaltx3    Tx3 Txsec3	Txother3	Txlean3	Txleansec3	SDM3	Aware3	Opinion3	Ask3	Info3	Explain3	Easyund3	Adv3	Disadv3	Decide3	Involve3	Satis3	Agree3	Discuss3	Satdec3	Sure3	Inform3	Clear3	Awaretx3	Informtx3	Decimp3	Mdtxrec3	Whatrec3	Strongrec3	Recdec3	Talkda3	Whoda3	mdrespme	irespmd	mdtime	recmd	Mdask3	Mdexp3	Mdcause3	Mdtalk3	Mdopin3	Iasktx3	Iaskrec3	Idetail3	Iaskque3	Isugtx3	Iinsist3	Idoubt3	Iopinion3	Die3	Survive3	Mdrec3	Imppart3	Avdsurg3	Avdpee3	Avdtime3	Avoided3	AvoidSexLevel	Avoidrec3	Mdrectx3	Mdrectxsec3	Txincon3	Txed3	Txfi3	Txridca3	Txnovec3	Likese3	Worryse3	Worrydie3	Sorryspr3	Ranked3	Rankpee3	Rankfi3	Rankrec3	Rankdie3	Death3	Dainf3	Dalook3	Dahelp3	Recda3	Darecog3	Daprep3	Daprocon3	Dapcimp3	Daorg3	Damatr3	Dainvo3	Daques3	Daprepmd3	Dafu3	Datrust3	Anx13	Anx23	Anx33	Anx43	Anx53	Anx63	Anx73	Anx83	Anx93	Anx103	Anx113	Anx123	Anx133	Anx143	Anx153	CRCTx3	MD_gender	MD_age	MD_race	MD_yrgrad	MD_type	MD_specialty	MD_specialtyother	MD_number_pts_wk	MD_percentpts	MD_number_noneng_pts	MD_white	MD_hispanic	MD_black	MD_native	MD_asian	MD_pacific	MD_raceother	MD_raceotherspecify'
temp_va_3_s = temp_va_3_s1 + "\t" + temp_va_3_s2
temp_va_3_l1 = temp_va_3_s.split("\t")
temp_va_3_l2 = []
for var in temp_va_3_l1:
    temp_va_3_l2.append('VA_' + var)
temp_va_3_l = temp_va_3_l1 + temp_va_3_l2

temp_va_4_s = 'txgot	TxgotT3prefcc	TxgotTx3cc'
temp_va_4_l1 = temp_va_4_s.split("\t")
temp_va_4_l2 = []
for var in temp_va_4_l1:
    temp_va_4_l2.append('VA_' + var)
temp_va_4_l = temp_va_4_l1 + temp_va_4_l2

temp_va_diagnostics_s = 'psa1   AvailTransc	risklevel	risklevel2	risklevel3'
temp_va_diagnostics_l1 = temp_va_diagnostics_s.split("\t")
temp_va_diagnostics_l2 = []
for var in temp_va_diagnostics_l1:
    temp_va_diagnostics_l2.append('VA_' + var)
temp_va_diagnostics_l = temp_va_diagnostics_l1 + temp_va_diagnostics_l2

temp_va_unknown_s = 'VAsite1    Ch_IL	RadOnc_apt	ActiveSurveillance	Surgery	Radiation	Brachy	Recom   TxStrongRec	PtReqRec	Ch3_comb	Vatrust1	Vatrust2	Vatrust3	Vatrust4	Vatrust5	Vatrust6	Vatrust7	Vatrust8	Vatrust9	Vatrust10	PreOp_apt	Tx_EndApt	b	Time2	VAsite2	daysT1T2	gleason	knowstage	stage	famhx	radonc	phonedx	Time3	MSIid	cmpt	interviewlength	hospital	batch	Transformations	SDMtri1	realmgrade	realmadeq	Numeracy	numeracydi	Anxiety1	SDMtri2	dietx2acc	dienot2acc	wait2acc	besttx2acc	sured2acc	raded2acc	wwed2acc	surpee2acc	radpee2acc	wwpee2acc	knowledge2	sharetot	DAsat	anxietyt2	SDMtri3	comradeall	comradesatcom	comradeconfdec	trustva	anxiety3	picsdocfac	picspatinfo	picspatdms	prepforDM	anxt2txpref	txconcur23	txconcur2got	txconcur3got	knowledge22	surgconcur23	beamconcur23	brachyconcur23	wwconcur23	surgconcur2got	beamconcur2got	brachyconcur2got	wwconcur2got	surgconcur3got	beamconcur3got	brachyconcur3got	wwconcur3got	die3dich	die3dich10	Avdsurg3dich	Avdsurg3dich10	avdpee3dich	avdpee3dich10	avdtime3dich	avdtime3dich10	avoided3dich	avoided3dich10	avoidrec3dich	avoidrec3dich10	filter_$	As_1	Ch3_AS'
temp_va_unknown_l1 = temp_va_unknown_s.split("\t")
temp_va_unknown_l2 = []
for var in temp_va_unknown_l1:
    temp_va_unknown_l2.append('VA_' + var)
temp_va_unknown_l = temp_va_unknown_l1 + temp_va_unknown_l2

#print(temp_va_unknown_l)

#creates the dictionary out of these lists
d_dvd = {1: temp_dvd_1_l, 2: temp_dvd_2_l, 3: temp_dvd_3_l, 4: temp_dvd_4_l, 'diagnostics': temp_dvd_diagnostics_l, 'unknown': temp_dvd_unknown_l}
d_va = {1: temp_va_1_l, 2: temp_va_2_l, 3: temp_va_3_l, 4: temp_va_4_l, 'diagnostics': temp_va_diagnostics_l, 'unknown': temp_va_unknown_l}

#diagnostic to check if dictionaries correctly made
#print(len(list(d_dvd.keys())))
#print(len(list(d_va.keys())))
#print('txgot' in d_va[3])

#print(temp_va_2_l)

#these are in the merged dataset and need to be sorted into stage
#print(['DVD_Act_t1_avg', 'DVD_Act_t2_avg', 'DVD_Act_t2_avg_mc', 'DVD_QOL1_Avghappy', 'DVD_QOL1_general', 'DVD_QOL1_health', 'DVD_QOL4_Avghappy', 'DVD_SDM2_dichot', 'DVD_Txother2', 'DVD_Unnamed: 0', 'DVD_data_num', 'DVD_databeyondt1', 'DVD_datat1', 'Txa2', 'Txb2', 'Txc2', 'VA_Anx102', 'VA_Anx122', 'VA_Anx132', 'VA_Anx22', 'VA_Anx32', 'VA_Anx42', 'VA_Anx82', 'VA_AvailTransc', 'VA_Besttx2', 'VA_Bring2', 'VA_Certainty_tx_EndApt', 'VA_Ch_IL', 'VA_Common2', 'VA_DA1', 'VA_Dahelp2', 'VA_Dahelptx2', 'VA_Dainf2', 'VA_Dalike2', 'VA_Dietx2', 'VA_MD', 'VA_Pbm2', 'VA_Radrec2', 'VA_Read2', 'VA_Recom', 'VA_SDM2', 'VA_Sexact2', 'VA_Sexcomp2', 'VA_Seximp2', 'VA_Sextx2', 'VA_Sharefam2', 'VA_Sharefri2', 'VA_Sharepar2', 'VA_Surgrec2', 'VA_TXother2', 'VA_TxDeferred', 'VA_TxStrongRec', 'VA_Txd2', 'VA_Txf2', 'VA_Txg2', 'VA_Txsec3', 'VA_Unnamed: 0', 'VA_VAsite1', 'VA_Whyww2', 'VA_psa2', 'VA_t3txpref', 'VA_whenrecruit', 'active_surv', 'decision3_vs_received', 'pref_treatment'])
