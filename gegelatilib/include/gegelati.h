/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2022) :
 *
 * Elinor Montmasson <elinor.montmasson@gmail.com> (2022)
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2022)
 * Mickaël Dardaillon <mdardail@insa-rennes.fr> (2022)
 * Nicolas Sourbier <nsourbie@insa-rennes.fr> (2019 - 2020)
 * Pierre-Yves Le Rolland-Raumer <plerolla@insa-rennes.fr> (2020)
 * Thomas Bourgoin <tbourgoi@insa-rennes.fr> (2021)
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

/**
 * \file gegelati.h
 * \brief Helper file gathering all headers from the GEGELATI lib to ease their
 * inclusion in apps.
 */
#ifndef GEGELATI_H
#define GEGELATI_H

#include <util/timestamp.h>

#include <data/array2DWrapper.h>
#include <data/arrayWrapper.h>
#include <data/constant.h>
#include <data/constantHandler.h>
#include <data/dataHandler.h>
#include <data/hash.h>
#include <data/pointerWrapper.h>
#include <data/primitiveTypeArray.h>
#include <data/primitiveTypeArray2D.h>
#include <data/untypedSharedPtr.h>

#include <file/parametersParser.h>
#include <file/tpgGraphDotExporter.h>
#include <file/tpgGraphDotImporter.h>

#include <instructions/addPrimitiveType.h>
#include <instructions/instruction.h>
#include <instructions/lambdaInstruction.h>
#include <instructions/multByConstant.h>
#include <instructions/set.h>

#include <learn/evaluationResult.h>
#include <learn/job.h>
#include <learn/learningAgent.h>
#include <learn/learningEnvironment.h>
#include <learn/learningParameters.h>
#include <learn/parallelLearningAgent.h>
#include <learn/CLagent.h>

#include <learn/adversarialEvaluationResult.h>
#include <learn/adversarialJob.h>
#include <learn/adversarialLearningAgent.h>
#include <learn/adversarialLearningEnvironment.h>

#include <learn/classificationEvaluationResult.h>
#include <learn/classificationLearningAgent.h>
#include <learn/classificationLearningEnvironment.h>

#include <log/cycleDetectionLALogger.h>
#include <log/laBasicLogger.h>
#include <log/laLogger.h>
#include <log/laPolicyStatsLogger.h>
#include <log/logger.h>

#include <mutator/lineMutator.h>
#include <mutator/mutationParameters.h>
#include <mutator/programMutator.h>
#include <mutator/rng.h>
#include <mutator/tpgMutator.h>

#include <program/line.h>
#include <program/program.h>
#include <program/programEngine.h>
#include <program/programExecutionEngine.h>

#include <tpg/policyStats.h>
#include <tpg/tpgAbstractEngine.h>
#include <tpg/tpgAction.h>
#include <tpg/tpgEdge.h>
#include <tpg/tpgExecutionEngine.h>
#include <tpg/tpgFactory.h>
#include <tpg/tpgGraph.h>
#include <tpg/tpgTeam.h>
#include <tpg/tpgVertex.h>

#include <tpg/instrumented/executionStats.h>
#include <tpg/instrumented/tpgActionInstrumented.h>
#include <tpg/instrumented/tpgEdgeInstrumented.h>
#include <tpg/instrumented/tpgExecutionEngineInstrumented.h>
#include <tpg/instrumented/tpgInstrumentedFactory.h>
#include <tpg/instrumented/tpgTeamInstrumented.h>
#include <tpg/instrumented/tpgVertexInstrumentation.h>

#ifdef CODE_GENERATION
#include <codeGen/programGenerationEngine.h>
#include <codeGen/tpgGenerationEngine.h>
#include <codeGen/tpgGenerationEngineFactory.h>
#include <codeGen/tpgStackGenerationEngine.h>
#include <codeGen/tpgSwitchGenerationEngine.h>
#endif

#include <archive.h>
#include <environment.h>

#endif
