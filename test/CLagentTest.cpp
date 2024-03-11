/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2023) :
 *
 * QuentinVacher <98522623+QuentinVacher-rl@users.noreply.github.com> (2023)
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2022)
 * Nicolas Sourbier <nsourbie@insa-rennes.fr> (2020)
 * Pierre-Yves Le Rolland-Raumer <plerolla@insa-rennes.fr> (2020)
 * Quentin Vacher <qvacher@insa-rennes.fr> (2023)
 *
 * GEGELATI is an open-source reinforcement learning framework for training
 * artificial intelligence based on Tangled Program Graphs (TPGs).
 *
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software. You can use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty and the software's author, the holder of the
 * economic rights, and the successive licensors have only limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading, using, modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean that it is complicated to manipulate, and that also
 * therefore means that it is reserved for developers and experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and, more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 */

#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>
#include <numeric>

#include "log/laBasicLogger.h"

#include "tpg/instrumented/tpgActionInstrumented.h"
#include "tpg/instrumented/tpgEdgeInstrumented.h"
#include "tpg/instrumented/tpgInstrumentedFactory.h"
#include "tpg/instrumented/tpgTeamInstrumented.h"
#include "tpg/instrumented/tpgVertexInstrumentation.h"
#include "tpg/policyStats.h"
#include "tpg/tpgGraph.h"

#include "instructions/addPrimitiveType.h"
#include "mutator/rng.h"
#include "mutator/tpgMutator.h"

#include "learn/learningAgent.h"
#include "learn/learningEnvironment.h"
#include "learn/learningParameters.h"
#include "learn/parallelLearningAgent.h"
#include "learn/stickGameWithOpponent.h"
#include "learn/CLagent.h"

class CLagentTest : public ::testing::Test
 {
   protected:
     Instructions::Set set;
     StickGameWithOpponent le;
     Learn::LearningParameters params;

     virtual void SetUp()
     {
         set.add(*(new Instructions::AddPrimitiveType<int>()));
         set.add(*(new Instructions::AddPrimitiveType<double>()));

         // Proba as in Kelly's paper
         params.mutation.tpg.maxInitOutgoingEdges = 3;
         params.mutation.prog.maxProgramSize = 96;
         params.mutation.tpg.nbRoots = 15;
         params.mutation.tpg.pEdgeDeletion = 0.7;
         params.mutation.tpg.pEdgeAddition = 0.7;
         params.mutation.tpg.pProgramMutation = 0.2;
         params.mutation.tpg.pEdgeDestinationChange = 0.1;
         params.mutation.tpg.pEdgeDestinationIsAction = 0.5;
         params.mutation.tpg.maxOutgoingEdges = 4;
         params.mutation.prog.pAdd = 0.5;
         params.mutation.prog.pDelete = 0.5;
         params.mutation.prog.pMutate = 1.0;
         params.mutation.prog.pSwap = 1.0;
         params.mutation.prog.pConstantMutation = 0.5;
         params.mutation.prog.minConstValue = 0;
         params.mutation.prog.maxConstValue = 1;
     }

     virtual void TearDown()
     {
         delete (&set.getInstruction(0));
         delete (&set.getInstruction(1));
     }
 };

 TEST_F(CLagentTest, Constructor)
 {
     Learn::CLagent* la;

     ASSERT_NO_THROW(la = new Learn::CLagent(le, set, params))
         << "Construction of the continuous learningAgent failed.";

     ASSERT_NO_THROW(delete la)
         << "Destruction of the continuous LearningAgent failed.";
 }

 TEST_F(CLagentTest, TrainOneagent)
 {
     params.archiveSize = 50;
     params.archivingProbability = 0.5;
     params.maxNbActionsPerEval = 11;
     params.nbIterationsPerPolicyEvaluation = 3;
     params.ratioDeletedRoots =
         0.95; // high number to force the apparition of root action.
     params.totalNbDel = 3;

     // we will validate in order to cover validation log
     params.doValidation = true;

     Learn::CLagent la(le, set, params);

     la.init();

     // we add a logger to la to check it logs things
     std::ofstream o("tempFileForTest", std::ofstream::out);
     Log::LABasicLogger l(la, o);

     // Do the populate call to keep know the number of initial vertex
     Archive a(0);
     Mutator::TPGMutator::populateTPG(*la.getTPGGraph(), a, params.mutation,
                                      la.getRNG(), 1);
     size_t initialNbVertex = la.getTPGGraph()->getNbVertices();
     // Seed selected so that an action becomes a root during next generation
     for (uint64_t i = 0; i < params.totalNbDel; ++i) {
         ASSERT_NO_THROW(la.trainOneAgent(4)) << "Training for one agent failed.";
     }
     // Check the number of vertex in the graph.
     // Must be initial number of vertex - number of root removed
     std::cout << "The value of totnbdel is: " << this->params.totalNbDel << std::endl;
     std::cout << "The value of a is: " << la.getTPGGraph()->getNbVertices() << std::endl;
     std::cout << "The value of b is: " << initialNbVertex << std::endl;
     std::cout << "The value of c is: " << params.ratioDeletedRoots << std::endl;
     std::cout << "The value of d is: " << params.mutation.tpg.nbRoots << std::endl;
     ASSERT_EQ(la.getTPGGraph()->getNbVertices(),
               initialNbVertex - floor(params.ratioDeletedRoots *
                                       params.mutation.tpg.nbRoots))
         << "Number of remaining is under the number of roots from the "
            "TPGGraph.";
     // Train a second generation, because most roots were removed, a root
     // actions have appeared and the training algorithm will attempt to remove
     // them.
     ASSERT_NO_THROW(la.trainOneAgent(0))
         << "Training for one agent failed.";

     // Check that bestScoreLastGen has been set
     ASSERT_NE(la.getBestScoreLastGen(), 0.0);

     // Check that bestRoot has been set
     ASSERT_NE(la.getBestRoot().first, nullptr);

     o.close();
     std::ifstream i("tempFileForTest", std::ofstream::in);
     std::string s;
     i >> s;
     ASSERT_TRUE(s.size() > 0) << "Logger should have logged elements after a "
                                  "trainOneAgent 'iteration'.";
     i.close();
     // removing the temporary file
     remove("tempFileForTest");
 }

 TEST_F(CLagentTest, TrainCL)
 {
     params.archiveSize = 50;
     params.archivingProbability = 0.5;
     params.maxNbActionsPerEval = 11;
     params.nbIterationsPerPolicyEvaluation = 5;
     params.ratioDeletedRoots = 0.2;
     params.nbGenerations = 3;

   Learn::CLagent la(le, set, params);

     la.init();
     bool alt = false;
     
     ASSERT_NO_THROW(la.trainCL(alt, true))
         << "Training a TPG for several generation should not fail.";
      alt = true;
     ASSERT_NO_THROW(la.trainCL(alt, true))
         << "Using the boolean reference to stop the training should not fail.";
 }

 TEST_F(CLagentTest, EvaluateAllRootsCL)
 {
     params.archiveSize = 50;
     params.archivingProbability = 0.5;
     params.maxNbActionsPerEval = 11;
     params.nbIterationsPerPolicyEvaluation = 10;

     Learn::CLagent la(le, set, params);

     la.init();
     std::multimap<std::shared_ptr<Learn::EvaluationResult>,
                   const TPG::TPGVertex*>
         result;
     ASSERT_NO_THROW(result =
                         la.evaluateAllRootsCL(0, Learn::LearningMode::TRAINING))
         << "Evaluation from a root failed.";
     ASSERT_EQ(result.size(), la.getTPGGraph()->getNbRootVertices())
         << "Number of evaluated roots is under the number of roots from the "
            "TPGGraph.";
 }

TEST_F(CLagentTest, EvalRootCL)
{
    params.archiveSize = 50;
    params.archivingProbability = 1.0;
    params.maxNbActionsPerEval = 11;
    params.nbIterationsPerPolicyEvaluation = 10;

    Learn::CLagent la(le, set, params);
    Archive a; // For testing purposes, notmally, the archive from the
    // LearningAgent is used.

    TPG::TPGExecutionEngine tee(la.getTPGGraph()->getEnvironment(), &a);

    la.init();
    std::shared_ptr<Learn::EvaluationResult> result;
    auto job = *la.makeJob(la.getTPGGraph()->getRootVertices().at(0),
                           Learn::LearningMode::TRAINING);
    ASSERT_NO_THROW(
        result = la.evaluateJobCL(tee, job, 0, Learn::LearningMode::TRAINING, le))
                    << "Evaluation from a root failed.";
    ASSERT_LE(result->getResult(), 1.0)
                    << "Average score should not exceed the score of a perfect player.";
}