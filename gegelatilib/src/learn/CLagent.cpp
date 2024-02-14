#include <inttypes.h>
#include <queue>

#include "data/hash.h"
#include "learn/evaluationResult.h"
#include "mutator/rng.h"
#include "mutator/tpgMutator.h"
#include "tpg/tpgExecutionEngine.h"


#include "learn/CLagent.h"

std::shared_ptr<Learn::EvaluationResult> Learn::CLagent::evaluateJobCL(
    TPG::TPGExecutionEngine& tee, const Job& job, uint64_t generationNumber,
    Learn::LearningMode mode, LearningEnvironment& le) const
{
    // Only consider the first root of jobs as we are not in adversarial mode
    const TPG::TPGVertex* root = job.getRoot();
    std::vector<double> previousScores;
    // Skip the root evaluation process if enough evaluations were already
    // performed. In the evaluation mode only.
    std::shared_ptr<Learn::EvaluationResult> previousEval;
    if (mode == LearningMode::TRAINING &&
        this->isRootEvalSkipped(*root, previousEval)) {
        return previousEval;
    }

    // Init results
    double result = 0.0;

    // Initialize action counter
    uint64_t totalActions = 0;

    // Initialize vars
    bool evalPassed = false;
    uint64_t bias = 1;
    uint64_t prevOutcome = 0;

    // Compute a Hash
    Data::Hash<uint64_t> hasher;
    uint64_t hash = hasher(generationNumber) ^ hasher(generationNumber);

    // Reset the learning Environment
    if (evalPassed == false) {
        le.reset(hash, mode, /*iterationNumber =*/0,generationNumber);
    }

    while (!le.isTerminal()) {
        // Get the action
        uint64_t actionID =
            ((const TPG::TPGAction*)tee.executeFromRoot(*root).back())
                ->getActionID();
        // Do it
        le.doAction(actionID);
        // Increment total actions
        totalActions++;

        // root does 'totalInteractions' amount of actions before passing to next root
        while (totalActions < this->params.totalInteractions) {
            prevOutcome += le.getScore();
            previousScores.push_back(prevOutcome);
        }

        if (previousScores.empty()) {
            std::cout << "Error: Cannot calculate average of an empty vector."
                      << std::endl;
        }

        uint64_t sum = 0.0;

        for (int i = 0; i < previousScores.size(); ++i) {
            sum += previousScores[i];
        }

        uint64_t average = sum / previousScores.size();

        // Check if it's time to perform an evaluation
        if (totalActions % this->params.totalInteractions == 0) {
            if (evalPassed == false) {
                prevOutcome += le.getScore();
            }
            if (le.getScore() > prevOutcome + 5 / 100 * prevOutcome) {
                bias = 1 + (le.getScore() / prevOutcome) / 10;
            }
            if (le.getScore() < prevOutcome - 5 / 100 * prevOutcome) {
                bias = 1 - (le.getScore() / prevOutcome) / 10;
            }
            else {
                bias = 1;
            }

            // Save previous score
            previousScores.push_back(prevOutcome);

            result += le.getScore() * bias;

            // modify previous scores depending on new score
            if (!previousScores.empty()) {
                if (le.getScore() > 800 &&
                    le.getScore() > previousScores.back())
                    previousScores.back() = previousScores.back() * 1.1;
            }
            if (le.getScore() < 200 && le.getScore() < previousScores.back())
                previousScores.back() = previousScores.back() * 0.9;
            prevOutcome = le.getScore();
            evalPassed = true;
        }
    }
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
    uint64_t generationNumber) // put into new class
{
    int nbdel = 0;
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
        this->evaluateAllRoots(generationNumber, LearningMode::TRAINING);
    nbdel++;
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
        nbdel = 0;
    }

    for (auto logger : loggers) {
        logger.get().logAfterDecimate();
    }

    // Does a validation or not according to the parameter doValidation
    if (params.doValidation) {
        auto validationResults =
            evaluateAllRoots(generationNumber, Learn::LearningMode::TRAINING);
        for (auto logger : loggers) {
            logger.get().logAfterValidate(validationResults);
        }
    }

    for (auto logger : loggers) {
        logger.get().logEndOfTraining();
    }
}

uint64_t Learn::CLagent::train(volatile bool& altTraining,
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