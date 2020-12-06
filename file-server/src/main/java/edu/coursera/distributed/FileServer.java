package edu.coursera.distributed;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;

/**
 * A basic and very limited implementation of a file server that responds to GET
 * requests from HTTP clients.
 */
public final class FileServer {
    /**
     * Main entrypoint for the basic file server.
     *
     * @param serverSocket Provided socket to accept connections on.
     * @param fs A proxy filesystem to serve files from. See the PCDPFilesystem
     *           class for more detailed documentation of its usage.
     * @throws IOException If an I/O error is detected on the server. This
     *                     should be a fatal error, your file server
     *                     implementation is not expected to ever throw
     *                     IOExceptions during normal operation.
     */
    public void run(final ServerSocket serverSocket, final PCDPFilesystem fs)
            throws IOException {
        /*
         * Enter a spin loop for handling client requests to the provided
         * ServerSocket object.
         */
        while (true) {
            Socket clientSocket = serverSocket.accept();

            /*
             * Using Socket.getInputStream(), parse the received HTTP
             * packet. In particular, we are interested in confirming this
             * message is a GET and parsing out the path to the file we are
             * GETing. Recall that for GET HTTP packets, the first line of the
             * received packet will look something like:
             *
             *     GET /path/to/file HTTP/1.1
             */
            InputStream inputStream = clientSocket.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));

            String line = reader.readLine();
            assert line != null;
            assert line.startsWith("GET"); // Validate GET request.
            final String filePath = line.split(" ")[1];
            PCDPPath fullPath = new PCDPPath(filePath);

            //System.out.println(fullPath.getAbsolutePath());


            /*
             * Using the parsed path to the target file, construct an
             * HTTP reply and write it to Socket.getOutputStream(). If the file
             * exists, the HTTP reply should be formatted as follows:
             *
             *   HTTP/1.0 200 OK\r\n
             *   Server: FileServer\r\n
             *   \r\n
             *   FILE CONTENTS HERE\r\n
             *
             * If the specified file does not exist, you should return a reply
             * with an error code 404 Not Found. This reply should be formatted
             * as:
             *
             *   HTTP/1.0 404 Not Found\r\n
             *   Server: FileServer\r\n
             *   \r\n
             *
             * Don't forget to close the output stream.
             */

            PrintWriter printer = new PrintWriter(clientSocket.getOutputStream());
            String content = fs.readFile(fullPath);

            if (content != null) {
                printer.write("HTTP/1.0 200 OK\r\n");
                printer.write("Server: FileServer\r\n");
                printer.write("\r\n");
                printer.write(content);
            } else {
                printer.write("HTTP/1.0 404 Not Found\r\n");
                printer.write("Server: FileServer\r\n");
                printer.write("\r\n");
            }

            printer.close();
            inputStream.close();
            reader.close();

        }
    }
}
