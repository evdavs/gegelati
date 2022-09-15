/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2022) :
 *
 * Emmanuel Montmasson <emontmas@insa-rennes.fr> (2022)
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2021)
 * Mickaël Dardaillon <mdardail@insa-rennes.fr> (2022)
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

#ifdef CODE_GENERATION

#ifndef TPG_STACK_GENERATION_ENGINE_H
#define TPG_STACK_GENERATION_ENGINE_H
#include "codeGen/tpgGenerationEngine.h"

namespace CodeGen {
    /**
     * \brief Class in charge of generating the C code of a TPGGraph.
     *
     * Each program of the TPGGraph is represented by a C function.
     * All the functions are regrouped in a file. Another file holds
     * the required functions to iterate through the TPGGraph.
     *
     * To use the generated code two code templates are provided in the
     * directory doc/codeGen. One template is for generic learning environment.
     * The other one is dedicated for adversarial learning environment and
     * manages the switch between the players. Both templates can use the
     * inference with the codeGen or the inference with Gegelati.
     *
     * The repo gegelati apps give some example of the template code completed
     * for TicTacToe, Pendulum and StickGame.
     */
    class TPGStackGenerationEngine : public CodeGen::TPGGenerationEngine
    {
      protected:
        /**
         * \brief function printing generic code in the main file.
         *
         * This function prints generic code to execute the TPG and manage the
         * stack of visited edges.
         */
        virtual void initTpgFile();

        /**
         * \brief function printing generic code declaration in the main file
         * header.
         *
         * This function print the the struct required to represent the TPG and
         * the prototypes of the function to execute the TPG and manage the
         * stack of visited edges.
         */
        virtual void initHeaderFile();

      public:
        /**
         * \brief Main constructor of the class.
         *
         * \param[in] filename : filename of the file holding the main function
         *                of the generated program.
         *
         * \param[in] tpg Environment in which the Program of the TPGGraph will
         *                be executed.
         *
         * \param[in] path to the folder in which the file are generated. If the
         * folder does not exist.
         */
        TPGStackGenerationEngine(const std::string& filename,
                                 const TPG::TPGGraph& tpg,
                                 const std::string& path = "./");

        /**
         * \brief destructor of the class.
         *
         * add endif at the end of the header and close both file.
         */
        ~TPGStackGenerationEngine();

        /**
         * \brief function that creates the C files required to execute the TPG
         * without gegelati.
         *
         * This function iterates trough the TPGGraph and create the required C
         * code to represent each element of the TPGGraph.
         */
        virtual void generateTPGGraph();

        /**
         * \brief Method for generating the code for an edge of the graph.
         *
         * This function generates the code that represents an edge.
         * An edge of a team is represented by a struct with:
         *  an integer,
         *  a function pointer of type : double (*ptr_prog)() for the program of
         * the edge
         *  a a function pointer of type : void* (*ptr_vertex)(int*) to
         * represent the destination of the edge
         *
         * \param[in] edge that must be generated.
         */
        virtual void generateEdge(const TPG::TPGEdge& edge);

        /**
         * \brief Method for generating the code for a team of the graph.
         *
         * This method generates the C function that represents a team.
         * Each function representing a team contains a static array of TPGEdge
         * and calls the function executeTeam(Edge*, int).
         *
         * \param[in] team const reference of the TPGTeam that must be
         * generated.
         */
        virtual void generateTeam(const TPG::TPGTeam& team);

        /**
         * \brief Method for generating a action of the graph.
         *
         * This method generates the C function that represents an action.
         * The generated function return a NULL pointer and write the action in
         * the pointer given as parameter.
         *
         * \param[in] action const reference of the TPGAction that must be
         * generated.
         */
        virtual void generateAction(const TPG::TPGAction& action);

        /**
         * \brief Define the function pointer root to the vertex given in
         * parameter.
         *
         * \param[in] root const reference to the root of the TPG graph.
         */
        virtual void setRoot(const TPG::TPGVertex& root);

        /**
         * \brief Generate function name depending on the vertex type.
         *
         * \param v vertex to be named.
         * \return std::string name of the vertex.
         */
        std::string vertexName(const TPG::TPGVertex& v);
    };
} // namespace CodeGen

#endif // TPG_STACK_GENERATION_ENGINE_H

#endif // CODE_GENERATION
