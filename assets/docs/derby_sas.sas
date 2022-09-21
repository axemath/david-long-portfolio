/****************************************************

Multivariate Analysis of Kentucky Derby Data
Binary Logistic Regression on Winner

By David Long
09.05.2020


****************************************************/

FILENAME odsout "&_SASWS_/derbyData2021/";
ODS HTML path=odsout style=dove
	body = "body.html"
	contents = "contents.html"
	frame = "index.html";
	
	
	
	
*Import data;

LIBNAME folder '/folders/myfolders/derbyData2021';

DATA work.derbyData;
	SET folder.derbydata;
	DROP bspd;
RUN;




*Check the contents of the derbyData dataset and print
the first 10 observations for verification;


TITLE "Derby Data Contents";
TITLE2 "Variables and Attributes";
PROC CONTENTS data=derbyData varnum;
RUN;

TITLE "Derby Winner Profiles";
PROC PRINT data=derbyData noobs;
	WHERE finish=1;
RUN;



/*
* Lifetime record;

DATA derbyData (drop=races--show);
	SET derbyData;
	pctWin = (win/races)*100;
	pctMoney = ((win + place + show)/races)*100;
RUN;


TITLE 'Lifetime Performance';
TITLE2 'of Winners';
PROC PRINT data=derbyData noobs;
	WHERE finish=1;
RUN;




* Graded stakes performance;

DATA derbyData (drop=gmoney--g1wins);
	SET derbyData;
	IF (gmoney^=0) THEN pctGWins = (gwins/gmoney)*100;
		ELSE pctGWins=0;
	IF (g1money^=0) THEN pctG1Wins = (g1wins/g1money)*100;
		ELSE pctG1Wins=0;
RUN;


TITLE 'Grades Stakes Performance';
TITLE2 'of Winners';
PROC PRINT data=derbyData noobs;
	WHERE finish=1;
RUN;
*/




* Winning Indicator and Missing Value Flag;

DATA derbyData;
	SET derbyData;
	
	* Binary success variable;
	IF finish=1 THEN success=1;  ELSE success=0;
	flag = 0;
	n = _N_;
		
	* flag missing values;
	ARRAY quant (*) _NUMERIC_;
	ARRAY qual (*) _CHARACTER_;
	DO i = 1 to dim(quant);
		IF quant(i)=. THEN flag=1;
	END;
	DO j = 1 to dim(qual);
		IF qual(j)=" " THEN flag=1;
	END;
	
	DROP i j;
	LABEL n = "ID Number";
RUN;


* Correlation with finish position;

TITLE 'Summary Statistics and';
TITLE2 'Correlation with Finish Position';
FOOTNOTE;
PROC CORR data=derbyData;
	VAR odds post--pp;
	WITH finish;
RUN;



/*
* Drop variables with low correlation;

DATA derbyData;
	SET derbyData (drop=post e1 e2);
RUN;
*/




*Binary logistic regression with stepwise selection of variables;

DATA logregr (drop=flag n);
	SET derbyData;
	WHERE flag=0;
RUN;


TITLE "Binary Logistic Regression";
TITLE2 "(Success = Win)";
PROC LOGISTIC data=logregr descending outmodel=outmodel;
	CLASS style;
	MODEL success = odds races--pp / lackfit;
	OUTPUT out=logregrpred p=p;
RUN;

TITLE "Probability of Winning";
TITLE2 "by Year";
PROC SORT data=logregrpred;
	BY year descending p;
RUN;

PROC PRINT data=logregrpred noobs label;
	VAR year name p finish odds;
	BY year;
RUN;

TITLE 'Instances Where the Model Was Correct';
DATA regSuccessStats;
	SET logregrpred;
	BY year;
	IF (finish=1) and first.year;
RUN;

PROC PRINT data=regSuccessStats label;
	ID name;
	VAR p finish year odds--pp;
RUN;

TITLE '2021 Field';
PROC PRINT data=folder.currentField label;
	ID name;
	VAR finish--pp;
RUN;

TITLE '2021 Predictions Based on the Model';
PROC LOGISTIC inmodel=outmodel;
	SCORE data=folder.currentField out=modelPred;
RUN;

PROC SORT data=modelPred;
	BY descending P_1;
RUN;

PROC PRINT data=modelPred label;
	VAR odds P_1 finish;
	ID name;
	LABEL name="Name" odds="Parimutuel Odds" P_1="Probability of Winning";
RUN;




* Horses with missing values;

TITLE 'Horses with Missing Values';
PROC PRINT data=derbyData;
	WHERE flag=1;
RUN;

ODS HTML CLOSE;