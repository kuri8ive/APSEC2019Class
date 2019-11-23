import net.arnx.jsonic.JSON;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * メソッド名と内部で呼び出しているメソッド名のリストのMapをJSON型に変換するクラス
 */
class JsonGenerator {

    private String tokenJson;

    JsonGenerator(HashMap<String, ArrayList<String>> results, boolean setPrettyPrint) {
        tokenJson = "";
        convertMethodDecIntoJson(results, setPrettyPrint);
    }

    /**
     * JSON型に変換する
     *
     * @param methodDecsHash keyがメソッド名を表す文字列、valueが内部で呼び出しているメソッド名を表す文字列のの配列となるようなHashMap
     * @param setPrettyPrint Trueの場合、改行・インデントを挿入し整形したJson文字列に変換する
     */
    private void convertMethodDecIntoJson(HashMap<String, ArrayList<String>> methodDecsHash, boolean setPrettyPrint) {
        HashMap<String, ArrayList<String>> tokenHash = new HashMap<>();

        for (HashMap.Entry<String, ArrayList<String>> entry : methodDecsHash.entrySet()) {
            String methodName = entry.getKey();
            tokenHash.put(methodName, entry.getValue());
        }
        JSON json = new JSON(JSON.Mode.STRICT);
        json.setPrettyPrint(true);
        tokenJson = json.format(tokenHash);
    }

    /**
     * 変換したJSON文字列を返す
     *
     * @return JSON文字列
     */
    String getJson() {
        return tokenJson;
    }

    /**
     * 変換したJSON文字列をファイルに出力する
     *
     * @param fileName 出力先ファイルパスを表す文字列
     * @throws IOException
     */
    void saveFile(String fileName) throws IOException {
        File file = new File(fileName);
        file.getParentFile().mkdirs();
        OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(file), "UTF-8");
        BufferedWriter bw = new BufferedWriter(osw);

        bw.write(tokenJson);

        bw.close();
    }

}
