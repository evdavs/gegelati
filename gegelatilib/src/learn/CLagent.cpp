#include <inttypes.h>
#include <queue>

#include "data/hash.h"
#include "learn/evaluationResult.h"
#include "mutator/rng.h"
#include "mutator/tpgMutator.h"
#include "tpg/tpgExecutionEngine.h"


#include "learn/CLagent.h"

double Learn::CLagent::calculateWeightDecay(double numScores) const
{
    // Example of a decay function that reaches zero and stays at zero
    if (numScores >= this->params.decayThreshold) {
        return 0.0; // Decay factor stays at zero after reaching a certain
                    // threshold
    }
    else {
        return (this->params.decayThreshold - numScores) / this->params.decayThreshold;
    }
}

std::shared_ptr<Learn::EvaluationResult> Learn::CLagent::evaluateJobCL(
    TPG::TPGExecutionEngine& tee, const Job& job, uint64_t generationNumber,
    Learn::LearningMode mode, LearningEnvironment& le)
{
    // Only consider the first root of jobs as we are not in adversarial mode
    const TPG::TPGVertex* root = job.getRoot();


    // Skip the root evaluation process if enough evaluations were already
    // performed. In the evaluation mode only.
    std::shared_ptr<Learn::EvaluationResult> previousEval;
    if (mode == LearningMode::TRAINING &&
        this->isRootEvalSkipped(*root, previousEval)) {
        return previousEval;
    }

    // Init results
    double result = 0.0;
    evalPassed = 0;
    // Initialize action counter
    uint64_t totalActions = 0;

    // Initialize vars

    double sum = 0.0;

    // Compute a Hash
    Data::Hash<uint64_t> hasher;
    uint64_t hash = hasher(generationNumber) ^ hasher(generationNumber);

    // Reset the learning Environment
        le.reset(hash, mode, /*iterationNumber =*/0,generationNumber);

    while (!le.isTerminal()) {
        // Get the action
        uint64_t actionID =
            ((const TPG::TPGAction*)tee.executeFromRoot(*root).back())
                ->getActionID();
        // Do it
        
        // Increment total actions
        

        // root does 'totalInteractions' amount of actions before passing to next root
/*        if (evalPassed == false) {
            while (totalActions < this->params.totalInteractions) {
                le.doAction(actionID);
                prevOutcome += le.getScore();
                previousScores.push_back(prevOutcome);
                totalActions++;
            }
            if (previousScores.empty()) {
                std::cout << "Error: Cannot calculate average of an empty "
                             "vector : division by zero"
                          << std::endl;
            }
            sum = 0.0;
            for (int i = 0; i < previousScores.size(); ++i) {
                sum += previousScores[i];   
            }
             
        }
        double result1 = sum / previousScores.size(); //*(1-influence) + infScoreAvg/this->params.decayThreshold;*/

//        double sizeRoot = previousScores.size();
            numScores = previousScores.size() - numScores;

            while (totalActions < this->params.totalInteractions) {
                le.doAction(actionID);
                if(!evalPassed){
                    prevOutcome += le.getScore();
                }
                else{
                    if(previousScores.back()){
                        double scoreFromLast = previousScores.back();
                        double lastScoreInf = calculateWeightDecay(numScores);
                        double lastScoreWeighted = scoreFromLast * lastScoreInf;
                        prevOutcome += le.getScore() * (1-lastScoreInf) + lastScoreWeighted;
                    }
                }
                previousScores.push_back(prevOutcome);
 /*               while (totalActions < this->params.decayThreshold){
                    earlyScores.push_back(le.getScore());
                }*/
                totalActions++;
 /*               double sumEarly = 0.0;
                for (int i = 0; i < earlyScores.size(); ++i) {
                    sumEarly += earlyScores[i] * (1-lastScoreInf) + lastScoreWeighted;
                }
                double avgEarly = sumEarly/earlyScores.size();*/
            }
            evalPassed = true;
            sum = 0.0;
            for (int i = 0; i < previousScores.size(); ++i) {
                sum += previousScores[i];
            }

            double result2 = sum / previousScores.size();

            result = result2;


    }
    // Create the EvaluationResult
    auto evaluationResult =
        std::shared_ptr<EvaluationResult>(new EvaluationResult(result,1));

    // Combine it with previous one if any
    if (previousEval != nullptr) {
        *evaluationResult += *previousEval;
    }
    return evaluationResult;
}

std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex*>
Learn::CLagent::evaluateAllRootsCL(uint64_t generationNumber,
    Learn::LearningMode mode)
{
    std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGVertex*>
        result;

    // Create the TPGExecutionEngine for this evaluation.
    // The engine uses the Archive only in training mode.
    std::unique_ptr<TPG::TPGExecutionEngine> tee =
        this->tpg->getFactory().createTPGExecutionEngine(
            this->env,
            (mode == LearningMode::TRAINING) ? &this->archive : NULL);

    auto roots = tpg->getRootVertices();
    for (int i = 0; i < roots.size(); i++) {
        auto job = makeJob(roots.at(i), mode);
        this->archive.setRandomSeed(job->getArchiveSeed());
        std::shared_ptr<EvaluationResult> avgScore = this->evaluateJobCL(
            *tee, *job, generationNumber, mode, this->learningEnvironment);
        result.emplace(avgScore, (*job).getRoot());
    }

    return result;
}

void Learn::CLagent::trainOneAgent(
    uint64_t generationNumber) 
{
    nbdel++;
    for (auto logger : loggers) {
        logger.get().logNewGeneration(generationNumber);
    }

    // Populate Sequentially
    Mutator::TPGMutator::populateTPG(*this->tpg, this->archive,
                                     this->params.mutation, this->rng,
                                     maxNbThreads);
    for (auto logger : loggers) {
        logger.get().logAfterPopulateTPG();
    }

    // Evaluate
    auto results =
        this->evaluateAllRootsCL(generationNumber, LearningMode::TRAINING);
    for (auto logger : loggers) {
        logger.get().logAfterEvaluate(results);
    }

    // Save the best score
    this->updateBestScoreLastGen(results);
    if (nbdel == this->params.totalNbDel) {
        // Remove worst performing roots
        decimateWorstRoots(results);
        // Update the best
        this->updateEvaluationRecords(results);
        nbdel =0;
    }

    for (auto logger : loggers) {
        logger.get().logAfterDecimate();
    }

    // Does a validation or not according to the parameter doValidation
    if (params.doValidation) {
        auto validationResults =
            evaluateAllRootsCL(generationNumber, Learn::LearningMode::TRAINING);
        for (auto logger : loggers) {
            logger.get().logAfterValidate(validationResults);
        }
    }

    for (auto logger : loggers) {
        logger.get().logEndOfTraining();
    }
}

uint64_t Learn::CLagent::trainCL(volatile bool& altTraining,
                                     bool printProgressBar)
{
    const int barLength = 50;
    uint64_t generationNumber = 0;

    while (!altTraining && generationNumber < this->params.nbGenerations) {
        // Train one generation
        trainOneAgent(generationNumber);
        generationNumber++;

        // Print progressBar (homemade, probably not ideal)
        if (printProgressBar) {
            printf("\rTraining ["); // back
            // filling ratio
            double ratio =
                (double)generationNumber / (double)this->params.nbGenerations;
            int filledPart = (int)((double)ratio * (double)barLength);
            // filled part
            for (int i = 0; i < filledPart; i++) {
                printf("%c", (char)219);
            }

            // empty part
            for (int i = filledPart; i < barLength; i++) {
                printf(" ");
            }

            printf("] %4.2f%%", ratio * 100.00);
        }
    }

    if (printProgressBar) {
        if (!altTraining) {
            printf("\nTraining completed\n");
        }
        else {
            printf("\nTraining alted at generation %" PRIu64 ".\n",
                   generationNumber);
        }
    }
    return generationNumber;
}