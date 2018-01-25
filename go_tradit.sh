DOMAIN=$1
EXP=$2

DATE=`date +"%m_%d_%y"`
TIME=`date +"%H_%M_%S"`

LOGDIR="logs/$DATE"
mkdir -p $LOGDIR

REPORTS_DIR="reports/$DATE"
mkdir -p $REPORTS_DIR

echo ""
echo "<TEST:>"
echo ""

for ITER in 1 2 3
do
for MAX_TRAIN in 30 60 300 1000 3000 8000 15000 20000

RUN="${DOMAIN}_${EXP}_${TIME}_${GOLD_EXP}_${ITER}"

DB_NAME="babble_${RUN}"
echo "Using db: $DB_NAME"
cp babble_${DOMAIN}_featurized_tocopy.db $DB_NAME.db

REPORTS_SUBDIR="$REPORTS_DIR/$RUN/"
mkdir -p $REPORTS_SUBDIR
echo "Saving reports to '$REPORTS_SUBDIR'"

LOGFILE="$LOGDIR/$RUN.log"
echo "Saving log to '$LOGFILE'"

python -u snorkel/contrib/babble/pipelines/run.py \
    --domain $DOMAIN \
    --reports_dir $REPORTS_SUBDIR \
    --db_name $DB_NAME \
    --start_at 7 \
    --end_at 10 \
    --supervision traditional \
    --disc_model_class logreg \
    --disc_model_search_space 10 \
    --verbose --no_plots |& tee -a $LOGFILE &
sleep 5

done
done
