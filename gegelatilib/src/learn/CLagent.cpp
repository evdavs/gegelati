#include <inttypes.h>
#include <queue>
#include <map>

#include "data/hash.h"
#include "learn/evaluationResult.h"
#include "mutator/rng.h"
#include "mutator/tpgMutator.h"
#include "tpg/tpgExecutionEngine.h"
#include "environment/pendulum.h"


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
    TPG::TPGExecutionEngine& tee, const Job& job, const Job& previousJob, uint64_t generationNumber,
    Learn::LearningMode mode, LearningEnvironment& le)
{
    bool evalPassed = false;
    // Only consider the first root of jobs as we are not in adversarial mode
    const TPG::TPGVertex* root = job.getRoot();

    // Skip the root evaluation process if enough evaluations were already
    // performed. In the evaluation mode only.
    std::shared_ptr<Learn::EvaluationResult> previousEval;
    if (mode == LearningMode::TRAINING &&
        this->isRootEvalSkipped(*root, previousEval)) {
        return previousEval;
    }

    // Init
    double result = 0.0;
    double prevOutcome =0.0;
    std::vector<double> previousScores;
    std::vector<double> earlyScores;
    std::vector<double> rootRes;
    double numScores = 0.0;

    // Initialize vars

    double sum;

    for (auto iterationNumber = 0; iterationNumber < this->params.nbIterationsPerPolicyEvaluation; iterationNumber++) {


            // Compute a Hash
            Data::Hash<uint64_t> hasher;
            uint64_t hash = hasher(generationNumber) ^ hasher(iterationNumber); //le.init(seed) en gros

            // Reset the learning Environment
            le.reset(hash, mode, iterationNumber, generationNumber);

        if (job.getIdx() > 0) {
            std::vector<stateEOE> retrievedVec = previousJob.getVecStateEOE();
            stateEOE desiredElement = retrievedVec[iterationNumber];
            double resetAngle = desiredElement.angle;
            double resetVelocity = desiredElement.velocity;
            ((Pendulum &) le).reset(hash, mode, iterationNumber, generationNumber, resetAngle, resetVelocity);

        }
        uint64_t nbActions = 0;
        while (!le.isTerminal() &&
               nbActions < this->params.maxNbActionsPerEval) {
            // Get the action
            uint64_t actionID =
                ((const TPG::TPGAction*)tee.executeFromRoot(*root).back())
                    ->getActionID();

 //           numScores = previousScores.size() - numScores;


            le.doAction(actionID);
            nbActions++;

 /*           for (int i = 0; i < this->params.maxNbActionsPerEval; ++i){
                previousScores.push_back(i*5);
            }*/
//            else{
    /*                      double scoreFromLast = previousScores.back();
                          double lastScoreInf = calculateWeightDecay(numScores);
                          double lastScoreWeighted = scoreFromLast *
               lastScoreInf;
                          prevOutcome += le.getScore() * (1-lastScoreInf) +
               lastScoreWeighted;
                          previousScores.push_back(prevOutcome);*/
 /*                         double sumEarly = 0.0;
                          while (nbActions < this->params.decayThreshold){
                              earlyScores.push_back(le.getScore());
                          }
                          for (int i = 0; i < earlyScores.size(); ++i) {
                              sumEarly += earlyScores[i] * (1-lastScoreInf) +
               lastScoreWeighted;
                          }
                          double avgEarly = sumEarly/earlyScores.size();
                          rootRes.back()+=avgEarly;*/

//            }
        }
        prevOutcome += le.getScore();
        //           if (previousScores.back() <= this->params.maxNbActionsPerEval) {
        previousScores.push_back(prevOutcome);
        //           }
        std::vector<stateEOE> vecPrev(this->params.nbIterationsPerPolicyEvaluation);
        vecPrev[iterationNumber].angle = ((Pendulum &) le).getAngle();
        vecPrev[iterationNumber].velocity = ((Pendulum &) le).getVelocity();
        job.setVecStateEOE(vecPrev);


//        rootRes.push_back(result);
    }
            sum = 0.0;
        for (int i = 0; i < previousScores.size(); ++i) {
            sum += previousScores[i];
        }
        result = sum;
/*
// Create the EvaluationResult
auto evaluationResult =
    std::shared_ptr<EvaluationResult>(new EvaluationResult(
        result / (double)params.nbIterationsPerPolicyEvaluation,
        params.nbIterationsPerPolicyEvaluation));

// Combine it with previous one if any
if (previousEval != nullptr) {
    *evaluationResult += *previousEval;
}
return evaluationResult;
*/

//    }
    // Create the EvaluationResult
//    if(!evalPassed) {
        auto evaluationResult = std::shared_ptr<EvaluationResult>(
            new EvaluationResult(result/ (double)params.nbIterationsPerPolicyEvaluation, params.nbIterationsPerPolicyEvaluation));
        // Combine it with previous one if any
        if (previousEval != nullptr) {
            *evaluationResult += *previousEval;
        }
        evalPassed = true;
        return evaluationResult;
 //  }
   /*   else{
        auto evaluationResult = std::shared_ptr<EvaluationResult>(
            new EvaluationResult(rootRes.end()[-1]/ (double)params.nbIterationsPerPolicyEvaluation, params.nbIterationsPerPolicyEvaluation));
        // Combine it with previous one if any
        if (previousEval != nullptr) {
            *evaluationResult += *previousEval;
        }
        evalPassed = true;
        return evaluationResult;
    }*/
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

    // Create a map to store IDs and corresponding vertices
    std::map<int, const TPG::TPGVertex*> vertexIDs;

    // Assign an int value to each root
    int index = 0;
    for (const auto& root : roots) {
        vertexIDs[index++] = root;
    }
    std::shared_ptr<Learn::Job> previousJob=nullptr;
    for (const auto& pair : vertexIDs) {
        auto job = makeJob(pair.second, mode, pair.first);
        this->archive.setRandomSeed(job->getArchiveSeed());
        std::shared_ptr<EvaluationResult> avgScore = this->evaluateJobCL(
            *tee, *job, *previousJob, generationNumber, mode, this->learningEnvironment);
        result.emplace(avgScore, (*job).getRoot());
        previousJob = job;
    }

    return result;
}

void Learn::CLagent::trainOneAgent(
    uint64_t generationNumber) 
{
    uint64_t nbdel =0;

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

    this->updateEvaluationRecords(results);
    for (auto logger : loggers) {
        logger.get().logAfterDecimate();
    }
    nbdel++;
//    if (nbdel == this->params.totalNbDel) {
        // Remove worst performing roots
        decimateWorstRoots(results);
        // Update the best
//    }
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