#ifndef CL_AGENT_H
#define CL_AGENT_H

#include <map>
#include <queue>

#include "archive.h"
#include "environment.h"
#include "instructions/set.h"
#include "log/laLogger.h"
#include "mutator/mutationParameters.h"
#include "tpg/tpgExecutionEngine.h"
#include "tpg/tpgGraph.h"

#include "learn/evaluationResult.h"
#include "learn/job.h"
#include "learn/learningEnvironment.h"
#include "learn/learningParameters.h"
#include "learn/learningAgent.h"

namespace Learn {

    /**
     * \brief Class used to control the learning steps of a TPGGraph within
     * a given LearningEnvironment.
     */
    class CLagent: public LearningAgent
    {
      private:
//        uint64_t nbdel = 0;
//        double prevOutcome = 0.0;


      public :


                  /**
         * \brief Constructor for LearningAgent.
         *
         * \param[in] le The LearningEnvironment for the TPG.
         * \param[in] iSet Set of Instruction used to compose Programs in the
         *            learning process.
         * \param[in] p The LearningParameters for the LearningAgent.
         * \param[in] factory The TPGFactory used to create the TPGGraph. A
         * default TPGFactory is used if none is provided.
         */
        CLagent(LearningEnvironment& le, const Instructions::Set& iSet,
                const LearningParameters& p,
                const TPG::TPGFactory& factory = TPG::TPGFactory())
                : LearningAgent(le, iSet, p, factory){}
        /**
* \brief Calculator for the weight decay.
*
* \param[in] numScores the number of scores obtained
         */
        double calculateWeightDecay(double numScores) const;
                        /**
         * \brief Train the TPGGraph for a given number of generation.
         *
         * The method trains the TPGGraph for a given number of generation,
         * unless the referenced boolean value becomes false (evaluated at each
         * generation).
         * Optionally, a simple progress bar can be printed within the terminal.
         * The TPGGraph is NOT (re)initialized before starting the training.
         *
         * \param[in] altTraining a reference to a boolean value that can be
         * used to halt the training process before its completion.
         * \param[in] printProgressBar select whether a progress bar will be
         * printed in the console. \return the number of completed generations.
         */
        uint64_t trainCL(volatile bool& altTraining, bool printProgressBar);
                
        /**
 * \brief Evaluates policy starting from the given root.
 *
 * The policy, that is, the TPGGraph execution starting from the given
 * TPGVertex is evaluated nbIteration times. The generationNumber is
 * combined with the current iteration number to generate a set of
 * seeds for evaluating the policy.
 *
 * The method is const to enable potential parallel calls to it.
 *
 * \param[in] tee The TPGExecutionEngine to use.
 * \param[in] job The job containing the root and archiveSeed for
 * the evaluation.
 * \param[in] previousJob The job containing the angles and vectors of previous job
 * \param[in] generationNumber the integer number of the current
 * generation.
 * \param[in] mode the LearningMode to use during the policy
 * evaluation.
 * \param[in] le Reference to the LearningEnvironment to use
 * during the policy evaluation (may be different from the attribute of
 * the class in child LearningAgentClass).
 *
 * \return a std::shared_ptr to the EvaluationResult for the root. If
 * this root was already evaluated more times then the limit in
 * params.maxNbEvaluationPerPolicy, then the EvaluationResult from the
 * resultsPerRoot map is returned, else the EvaluationResult of the
 * current generation is returned, already combined with the
 * resultsPerRoot for this root (if any).
 */
        std::shared_ptr<EvaluationResult> evaluateJobCL(
            TPG::TPGExecutionEngine& tee, const Job& job,
            const Job& previousJob,
            uint64_t generationNumber, LearningMode mode,
            LearningEnvironment& le);

                        /**
         * \brief Train the TPGGraph for one agent.
         *
         * Training for one agent includes:
         * - Populating the TPGGraph according to given MutationParameters.
         * - Evaluating all roots of the TPGGraph. (call to evaluateAllRoots and
         * modified EvaluateJob)
         * - Removing from the TPGGraph the worst root (after a while)
         *
         * \param[in] generationNumber the integer number of the current
         * generation.
         */
        void trainOneAgent(uint64_t generationNumber);

        /**
         * \brief Evaluate all root TPGVertex of the TPGGraph.
         *
         * This method calls the evaluateJob method for every root TPGVertex
         * of the TPGGraph. The method returns a sorted map associating each
         * root vertex to its average score, in ascending order or score.
         *
         * \param[in] generationNumber the integer number of the current
         * generation.
         * \param[in] mode the LearningMode to use during the policy
         * evaluation.
         */
        std::multimap<std::shared_ptr<EvaluationResult>,
                              const TPG::TPGVertex*>
        evaluateAllRootsCL(uint64_t generationNumber, LearningMode mode);

    };
}; // namespace Learn
#endif
